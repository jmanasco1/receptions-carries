"""A thin, credit-aware client for The Odds API v4.

Credit accounting is the point of this module, not an add-on. The free tier is
500 credits/month and a single full-slate props pull costs roughly
`events x markets x regions` (~32 for a 16-game slate, 2 markets, 1 region).
It is entirely possible to burn a month's quota in an afternoon of debugging,
so:

  * every response's `x-requests-remaining` / `x-requests-used` /
    `x-requests-last` headers are parsed and written to the metadata DB;
  * the client hard-aborts before a call when the last known remaining balance
    is below the configured floor;
  * `get_event_odds_bulk` refuses to start a pull whose estimated cost exceeds
    `max_spend_per_run`, so a slate that unexpectedly has 40 events cannot
    silently drain the quota.

Retries use exponential backoff on 429 and 5xx only. A 401/422 is a bug in our
request and retrying it just wastes credits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from nfl_usage_props.config import Config
from nfl_usage_props.metadata import MetadataStore

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OddsAPIError(RuntimeError):
    """Any non-recoverable failure talking to The Odds API."""


class CreditFloorError(OddsAPIError):
    """Raised when a call would run the credit balance below the floor."""


@dataclass
class OddsResponse:
    """A successful response plus the credit headers that came with it."""

    data: Any
    status_code: int
    requests_remaining: int | None = None
    requests_used: int | None = None
    cost: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    fetched_at: str = ""


def _parse_int_header(headers: Any, name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def estimate_props_cost(n_events: int, n_markets: int, n_regions: int = 1) -> int:
    """Credits a per-event props pull will cost.

    The Odds API charges the event-odds endpoint as markets x regions *per
    event*. Documented behaviour; see README for the verification status.
    """
    return max(0, n_events) * max(1, n_markets) * max(1, n_regions)


class OddsAPIClient:
    def __init__(
        self,
        config: Config,
        *,
        meta: MetadataStore | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
        sleep: Any = time.sleep,
    ):
        self.config = config
        self.odds_cfg = config.odds
        self.meta = meta if meta is not None else MetadataStore(config.metadata_db)
        self._api_key = api_key if api_key is not None else config.odds_api_key()
        self.session = session or requests.Session()
        self._sleep = sleep
        # Populated from response headers; None until the first call.
        self.requests_remaining: int | None = self.meta.latest_credit_balance()
        self.spent_this_run: int = 0

    # ------------------------------------------------------------------ guards

    def _require_key(self) -> str:
        if not self._api_key:
            raise OddsAPIError(
                "ODDS_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self._api_key

    def _check_floor(self, *, about_to_spend: int = 1) -> None:
        floor = self.odds_cfg.credits.floor
        remaining = self.requests_remaining
        if remaining is None:
            return  # No balance observed yet; the first call establishes it.
        if remaining - about_to_spend < floor:
            raise CreditFloorError(
                f"refusing to call: {remaining} credits remaining, floor is {floor}, "
                f"this call costs ~{about_to_spend}"
            )

    # ----------------------------------------------------------------- request

    def _request(
        self, path: str, params: dict[str, Any], *, expected_cost: int = 1
    ) -> OddsResponse:
        self._check_floor(about_to_spend=expected_cost)

        url = f"{self.odds_cfg.base_url}{path}"
        full_params = {"apiKey": self._require_key(), **params}
        # Never log the key.
        safe_params = json.dumps({k: v for k, v in params.items()}, sort_keys=True)

        retry = self.odds_cfg.retry
        delay = retry.backoff_initial_seconds
        last_error: str | None = None

        for attempt in range(1, retry.max_attempts + 1):
            started = time.perf_counter()
            try:
                resp = self.session.get(url, params=full_params, timeout=retry.timeout_seconds)
            except requests.RequestException as exc:
                duration = time.perf_counter() - started
                last_error = f"{type(exc).__name__}: {exc}"
                self.meta.record_odds_call(
                    endpoint=path, params=safe_params, duration_s=duration, error=last_error
                )
                if attempt == retry.max_attempts:
                    raise OddsAPIError(
                        f"request failed after {attempt} attempts: {last_error}"
                    ) from exc
                self._sleep(delay)
                delay = min(delay * retry.backoff_multiplier, retry.backoff_max_seconds)
                continue

            duration = time.perf_counter() - started
            remaining = _parse_int_header(resp.headers, "x-requests-remaining")
            used = _parse_int_header(resp.headers, "x-requests-used")
            cost = _parse_int_header(resp.headers, "x-requests-last")

            if remaining is not None:
                self.requests_remaining = remaining
            if cost is not None:
                self.spent_this_run += cost

            error = None if resp.ok else f"HTTP {resp.status_code}: {resp.text[:500]}"
            self.meta.record_odds_call(
                endpoint=path,
                params=safe_params,
                status_code=resp.status_code,
                requests_used=used,
                requests_remaining=remaining,
                cost=cost,
                duration_s=duration,
                error=error,
            )

            if resp.ok:
                return OddsResponse(
                    data=resp.json(),
                    status_code=resp.status_code,
                    requests_remaining=remaining,
                    requests_used=used,
                    cost=cost,
                    headers=dict(resp.headers),
                    fetched_at=datetime.now(UTC).isoformat(),
                )

            last_error = error
            if resp.status_code not in RETRYABLE_STATUS:
                # 401 (bad key), 404, 422 (bad market key) -- retrying burns
                # credits without any chance of succeeding.
                raise OddsAPIError(last_error or f"HTTP {resp.status_code}")

            if attempt == retry.max_attempts:
                break
            self._sleep(delay)
            delay = min(delay * retry.backoff_multiplier, retry.backoff_max_seconds)

        raise OddsAPIError(f"request failed after {retry.max_attempts} attempts: {last_error}")

    # --------------------------------------------------------------- endpoints

    def get_sports(self) -> OddsResponse:
        """List in-season sports. Free -- does not consume credits."""
        return self._request("/sports", {}, expected_cost=0)

    def get_events(self) -> OddsResponse:
        """List upcoming events (ids, teams, commence times). Free."""
        return self._request(f"/sports/{self.odds_cfg.sport}/events", {}, expected_cost=0)

    def get_game_odds(self, markets: list[str] | None = None) -> OddsResponse:
        """Spreads and totals for the whole slate. Costs markets x regions."""
        markets = markets or list(self.odds_cfg.game_markets)
        n_regions = len(self.odds_cfg.regions.split(","))
        return self._request(
            f"/sports/{self.odds_cfg.sport}/odds",
            {
                "markets": ",".join(markets),
                "regions": self.odds_cfg.regions,
                "oddsFormat": self.odds_cfg.odds_format,
            },
            expected_cost=len(markets) * n_regions,
        )

    def get_event_odds(self, event_id: str, markets: list[str] | None = None) -> OddsResponse:
        """Player props for ONE event. Costs markets x regions."""
        markets = markets or list(self.odds_cfg.prop_markets)
        n_regions = len(self.odds_cfg.regions.split(","))
        return self._request(
            f"/sports/{self.odds_cfg.sport}/events/{event_id}/odds",
            {
                "markets": ",".join(markets),
                "regions": self.odds_cfg.regions,
                "oddsFormat": self.odds_cfg.odds_format,
            },
            expected_cost=len(markets) * n_regions,
        )

    def get_event_odds_bulk(
        self,
        event_ids: list[str],
        markets: list[str] | None = None,
        *,
        on_event: Any = None,
    ) -> list[OddsResponse]:
        """Props for a list of events, with a pre-flight budget check.

        Aborts before the first call if the whole pull would exceed
        `max_spend_per_run`, and again mid-pull if the balance drops below the
        floor -- a partial slate is a far better outcome than an exhausted
        quota.
        """
        markets = markets or list(self.odds_cfg.prop_markets)
        n_regions = len(self.odds_cfg.regions.split(","))
        estimate = estimate_props_cost(len(event_ids), len(markets), n_regions)
        budget = self.odds_cfg.credits.max_spend_per_run
        if estimate > budget:
            raise CreditFloorError(
                f"estimated cost {estimate} credits for {len(event_ids)} events exceeds "
                f"max_spend_per_run={budget}"
            )

        responses: list[OddsResponse] = []
        for event_id in event_ids:
            resp = self.get_event_odds(event_id, markets)
            responses.append(resp)
            if on_event:
                on_event(event_id, resp)
        return responses

    # ------------------------------------------------------------------ saving

    def save_snapshot(self, response: OddsResponse, name: str) -> Path:
        """Persist a raw response verbatim, timestamped.

        Raw snapshots are the only historical prop record we will ever have --
        The Odds API's historical endpoints are paid-tier. Never overwrite one.
        """
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.config.odds_dir / name / f"{stamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": response.fetched_at,
            "status_code": response.status_code,
            "requests_remaining": response.requests_remaining,
            "requests_used": response.requests_used,
            "cost": response.cost,
            "data": response.data,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path
