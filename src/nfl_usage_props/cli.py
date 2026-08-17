"""Command line interface."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from nfl_usage_props.config import load_config
from nfl_usage_props.ingest import TABLES, IngestResult, ingest_all
from nfl_usage_props.metadata import MetadataStore
from nfl_usage_props.odds import CreditFloorError, OddsAPIClient, OddsAPIError
from nfl_usage_props.storage import ParquetStore

app = typer.Typer(help="NFL usage props: receptions and rush attempts.", no_args_is_help=True)
ingest_app = typer.Typer(help="nflverse data ingestion.", no_args_is_help=True)
odds_app = typer.Typer(help="The Odds API client.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(odds_app, name="odds")

console = Console()


def _parse_seasons(spec: str | None) -> list[int] | None:
    """Accept '2023', '2020-2024', or '2019,2021,2023'."""
    if not spec:
        return None
    seasons: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            seasons.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            seasons.append(int(chunk))
    return sorted(set(seasons))


@ingest_app.command("run")
def ingest_run(
    tables: Annotated[
        str | None, typer.Option("--tables", help="Comma-separated table names.")
    ] = None,
    seasons: Annotated[
        str | None, typer.Option("--seasons", help="e.g. 2023 or 2016-2024 or 2019,2023")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download even completed seasons.")
    ] = False,
) -> None:
    """Pull nflverse tables to parquet. Idempotent -- completed seasons are skipped."""
    config = load_config()
    table_list = [t.strip() for t in tables.split(",")] if tables else None

    def report(result: IngestResult) -> None:
        if result.status == "fetched":
            console.print(
                f"[green]fetched[/] {result.table} {result.season} "
                f"rows={result.rows:,} {result.bytes_written / 1e6:.1f}MB "
                f"{result.duration_s:.1f}s"
            )
        elif result.status == "skipped_complete":
            console.print(f"[dim]skipped[/] {result.table} {result.season} (complete)")
        elif result.status == "unavailable":
            console.print(
                f"[yellow]pending[/] {result.table} {result.season} (not published upstream yet)"
            )
        else:
            console.print(f"[red]error[/]   {result.table} {result.season}: {result.message}")

    results = ingest_all(
        config, tables=table_list, seasons=_parse_seasons(seasons), force=force, progress=report
    )

    fetched = sum(1 for r in results if r.status == "fetched")
    skipped = sum(1 for r in results if r.status == "skipped_complete")
    pending = sum(1 for r in results if r.status == "unavailable")
    errors = [r for r in results if r.status == "error"]
    console.print(
        f"\n[bold]{fetched} fetched, {skipped} skipped, {pending} pending, "
        f"{len(errors)} errors[/] -> {config.raw_dir}"
    )
    # `pending` is not a failure: a future season has no game data by definition.
    if errors:
        raise typer.Exit(code=1)


@ingest_app.command("status")
def ingest_status() -> None:
    """Show what is on disk, by table and season."""
    config = load_config()
    store = ParquetStore(config.raw_dir, compression=config.storage.compression)
    meta = MetadataStore(config.metadata_db)
    states = {(r["table_name"], r["season"]): r for r in meta.ingest_summary()}

    table = Table(title=f"Ingested data — {config.raw_dir}")
    for col in ("table", "seasons on disk", "complete", "total rows", "size"):
        table.add_column(col)

    for name in sorted(TABLES):
        seasons = store.available_seasons(name)
        if not seasons:
            table.add_row(name, "[dim]none[/]", "-", "-", "-")
            continue
        rows = sum((states.get((name, s), {}).get("rows") or 0) for s in seasons)
        size = sum(store.size_bytes(name, s) for s in seasons)
        complete = sum(1 for s in seasons if (states.get((name, s), {}).get("complete")))
        span = f"{min(seasons)}–{max(seasons)} ({len(seasons)})"
        table.add_row(name, span, f"{complete}/{len(seasons)}", f"{rows:,}", f"{size / 1e6:.1f}MB")

    console.print(table)


@odds_app.command("credits")
def odds_credits() -> None:
    """Show the last observed credit balance and recent calls. Makes no API call."""
    config = load_config()
    meta = MetadataStore(config.metadata_db)
    balance = meta.latest_credit_balance()
    floor = config.odds.credits.floor

    if balance is None:
        console.print("[yellow]No credit balance recorded yet[/] (no API call has been made).")
    else:
        colour = "red" if balance < floor else "green"
        console.print(f"Last observed balance: [{colour}]{balance}[/] (floor {floor})")

    calls = meta.recent_odds_calls(limit=10)
    if not calls:
        console.print("[dim]No recorded API calls.[/]")
        return
    table = Table(title="Recent Odds API calls")
    for col in ("when", "endpoint", "status", "cost", "remaining", "error"):
        table.add_column(col, overflow="fold")
    for call in calls:
        table.add_row(
            (call["created_at"] or "")[:19],
            call["endpoint"] or "",
            str(call["status_code"] or ""),
            str(call["cost"] if call["cost"] is not None else ""),
            str(call["requests_remaining"] if call["requests_remaining"] is not None else ""),
            (call["error"] or "")[:60],
        )
    console.print(table)


@odds_app.command("verify-markets")
def odds_verify_markets(
    save_fixture: Annotated[
        bool, typer.Option("--save-fixture", help="Write the response to tests/fixtures/.")
    ] = False,
) -> None:
    """Verify the configured prop market keys against the live API.

    Costs credits: one free /events call, then ONE event-odds call
    (markets x regions credits, ~2 by default). This is the single real call
    Stage 1 is designed around -- run it once, capture the fixture, and work
    offline from there.
    """
    config = load_config()
    client = OddsAPIClient(config)

    try:
        events = client.get_events()
    except (OddsAPIError, CreditFloorError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    event_list = events.data or []
    console.print(f"{len(event_list)} upcoming events. Remaining: {events.requests_remaining}")
    if not event_list:
        console.print("[yellow]No upcoming NFL events — nothing to verify (offseason?).[/]")
        raise typer.Exit(code=1)

    event = event_list[0]
    console.print(
        f"Probing event [bold]{event.get('away_team')} @ {event.get('home_team')}[/] "
        f"({event.get('id')})"
    )

    try:
        resp = client.get_event_odds(event["id"])
    except (OddsAPIError, CreditFloorError) as exc:
        console.print(f"[red]{exc}[/]")
        console.print(
            "[yellow]A 422 here usually means a market key is wrong. "
            "Check config.toml [odds].prop_markets.[/]"
        )
        raise typer.Exit(code=1) from exc

    found = {
        m.get("key")
        for book in (resp.data or {}).get("bookmakers", [])
        for m in book.get("markets", [])
    }
    console.print(f"Cost {resp.cost} credits. Remaining: {resp.requests_remaining}")
    console.print(f"Bookmakers: {len((resp.data or {}).get('bookmakers', []))}")

    for market in config.odds.prop_markets:
        mark = "[green]FOUND[/]" if market in found else "[red]MISSING[/]"
        console.print(f"  {mark} {market}")

    saved = client.save_snapshot(resp, "verify")
    console.print(f"Snapshot: {saved}")

    if save_fixture:
        from nfl_usage_props.config import REPO_ROOT

        fixture = REPO_ROOT / "tests" / "fixtures" / "event_odds_live.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(resp.data, indent=2))
        console.print(f"Fixture: {fixture}")

    if not set(config.odds.prop_markets) <= found:
        raise typer.Exit(code=1)


@odds_app.command("snapshot")
def odds_snapshot(
    kind: Annotated[str, typer.Option("--kind", help="open | close | repoll")] = "open",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be pulled, spend nothing.")
    ] = False,
) -> None:
    """Log a prop snapshot. This is the CLV record — run it on schedule.

    Costs markets x regions per event (~2 each). "close" only picks up events
    that have not yet passed the cutoff, so it is safe to run on every game day
    and will never pay for a game already in progress.
    """
    from datetime import UTC, datetime

    from nfl_usage_props.odds.snapshots import events_to_snapshot, take_snapshot

    config = load_config()
    client = OddsAPIClient(config)

    if dry_run:
        try:
            events = client.get_events().data or []
        except (OddsAPIError, CreditFloorError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        selected = events_to_snapshot(
            events,
            now=datetime.now(UTC),
            horizon_hours=config.odds.snapshots.horizon_hours,
            close_cutoff_minutes=config.odds.snapshots.close_cutoff_minutes,
        )
        cost = len(selected) * len(config.odds.prop_markets)
        console.print(
            f"[bold]{len(selected)}[/] of {len(events)} events in window, "
            f"estimated [bold]{cost}[/] credits"
        )
        for event in selected:
            console.print(
                f"  {event.get('commence_time')}  "
                f"{event.get('away_team')} @ {event.get('home_team')}"
            )
        return

    try:
        result = take_snapshot(client, config, kind=kind)
    except (OddsAPIError, CreditFloorError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]{result.snapshot_id}[/] {result.events_fetched}/{result.events_requested} events, "
        f"{result.rows:,} rows, {result.credits_spent} credits spent, "
        f"{result.credits_remaining} remaining"
    )
    if result.path:
        console.print(f"  {result.path}")
    for error in result.errors:
        console.print(f"  [yellow]{error}[/]")
    if result.errors and result.events_fetched == 0:
        raise typer.Exit(code=1)


@odds_app.command("resolve")
def odds_resolve(
    strict: Annotated[
        bool,
        typer.Option("--strict/--report", help="Raise on unresolved names, or just list them."),
    ] = False,
) -> None:
    """Map book player names in logged snapshots to gsis_ids.

    Runs over snapshots already on disk, so it is safe to fail: nothing is lost
    by raising here, and the pass can be re-run after adding overrides.
    """
    import polars as pl

    from nfl_usage_props.identity import PlayerResolver
    from nfl_usage_props.odds.snapshots import SnapshotStore, resolve_snapshot
    from nfl_usage_props.reference import load_name_overrides
    from nfl_usage_props.storage import SEASONLESS_SENTINEL

    config = load_config()
    store = ParquetStore(config.raw_dir)
    if not store.exists("players", SEASONLESS_SENTINEL):
        console.print("[red]No players table. Run `nfl-props ingest run` first.[/]")
        raise typer.Exit(code=1)

    snapshots = SnapshotStore(config.props_dir).read_all()
    if snapshots.is_empty():
        console.print("[yellow]No snapshots logged yet.[/]")
        raise typer.Exit(code=1)

    players = store.read("players", SEASONLESS_SENTINEL)
    if "latest_team" in players.columns:
        players = players.with_columns(pl.col("latest_team").alias("team"))

    team_lookup: dict[str, str] = {}
    if store.exists("teams", SEASONLESS_SENTINEL):
        teams = store.read("teams", SEASONLESS_SENTINEL)
        team_lookup = dict(zip(teams["team_name"], teams["team_abbr"], strict=False))

    resolver = PlayerResolver(players, overrides=load_name_overrides())
    console.print(f"Resolver universe: {resolver.universe_size:,} skill players")

    resolved, unresolved = resolve_snapshot(
        snapshots, resolver, team_lookup=team_lookup, strict=False
    )
    total = snapshots["player_name_raw"].n_unique()
    matched = resolved.filter(pl.col("gsis_id").is_not_null())["player_name_raw"].n_unique()

    console.print(f"[green]{matched}[/]/{total} distinct names resolved")
    for message in unresolved:
        console.print(f"  [yellow]{message}[/]")

    if unresolved and strict:
        raise typer.Exit(code=1)


@odds_app.command("fetch-game")
def odds_fetch_game() -> None:
    """Pull spreads and totals for the whole slate (2 credits) and snapshot it."""
    config = load_config()
    client = OddsAPIClient(config)
    try:
        resp = client.get_game_odds()
    except (OddsAPIError, CreditFloorError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    path = client.save_snapshot(resp, "game_odds")
    console.print(
        f"[green]{len(resp.data)} games[/] cost={resp.cost} "
        f"remaining={resp.requests_remaining}\n{path}"
    )


docs_app = typer.Typer(help="Generated documentation.", no_args_is_help=True)
app.add_typer(docs_app, name="docs")


@docs_app.command("data-dictionary")
def docs_data_dictionary(
    out: Annotated[str, typer.Option("--out", help="Output path.")] = "docs/DATA_DICTIONARY.md",
) -> None:
    """Regenerate the data dictionary from the parquet currently on disk."""
    from pathlib import Path

    from nfl_usage_props.config import REPO_ROOT
    from nfl_usage_props.datadict import build, render

    config = load_config()
    docs = build(config)
    if not docs:
        console.print("[red]No ingested tables found. Run `nfl-props ingest run` first.[/]")
        raise typer.Exit(code=1)

    path = Path(out)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(config, docs))

    tables = len({d.table for d in docs})
    console.print(f"[green]{len(docs)} columns across {tables} tables[/] -> {path}")


@app.command("config")
def show_config() -> None:
    """Print the resolved configuration."""
    config = load_config()
    console.print(f"[bold]config[/]      {config.source_path}")
    console.print(f"[bold]data_dir[/]    {config.data_dir}")
    console.print(
        f"[bold]seasons[/]     {config.ingest.seasons()[0]}–{config.ingest.seasons()[-1]}"
    )
    console.print(f"[bold]tables[/]      {', '.join(config.ingest.tables)}")
    console.print(f"[bold]prop mkts[/]   {', '.join(config.odds.prop_markets)}")
    console.print(f"[bold]credit floor[/] {config.odds.credits.floor}")
    key = config.odds_api_key()
    console.print(f"[bold]ODDS_API_KEY[/] {'set' if key else '[yellow]not set[/]'}")


if __name__ == "__main__":
    app()


features_app = typer.Typer(help="Stage 2: the as-of feature matrix.", no_args_is_help=True)
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    seasons: Annotated[
        str | None, typer.Option("--seasons", help="e.g. 2024 or 2020-2024.")
    ] = None,
    write: Annotated[
        bool, typer.Option("--write/--no-write", help="Persist to data/derived.")
    ] = True,
) -> None:
    """Build the player-game panel and the as-of feature matrix."""
    from nfl_usage_props.features.build import feature_columns
    from nfl_usage_props.features.dataset import build_dataset, write_dataset

    config = load_config()
    panel, features = build_dataset(config, _parse_seasons(seasons))

    console.print(f"[bold]panel[/]     {panel.height:,} player-games")
    console.print(
        f"[bold]features[/]  {features.height:,} rows, {len(feature_columns(features))} features"
    )

    covered = sorted(panel["season"].unique().to_list())
    console.print(f"[bold]seasons[/]   {covered[0]}–{covered[-1]}")

    if write:
        written = write_dataset(config, panel, features)
        console.print(f"[green]wrote {len(written)} partitions[/] -> {config.derived_dir}")


@features_app.command("leakage")
def features_leakage(
    season: Annotated[int, typer.Option("--season", help="Season to test.")],
    weeks: Annotated[str, typer.Option("--weeks", help="e.g. 4 or 1,8,14.")] = "4,10",
) -> None:
    """Re-derive the feature matrix from a corrupted and a truncated corpus.

    Any feature that moves is reading either its own game's outcome or a game
    that had not been played yet. Exits non-zero on any finding, so this is
    usable as a gate rather than as a report nobody reads.
    """
    from nfl_usage_props.features.dataset import load_panel, load_schedules
    from nfl_usage_props.leakage import run_leakage_checks

    config = load_config()
    panel = load_panel(config)
    schedules = load_schedules(config)

    findings = []
    for week in _parse_seasons(weeks) or []:
        found = run_leakage_checks(panel, schedules, config, season=season, week=week)
        status = "[green]clean[/]" if not found else f"[red]{len(found)} findings[/]"
        console.print(f"{season} week {week:>2}: {status}")
        findings.extend(found)

    if findings:
        for finding in findings:
            console.print(f"  [red]{finding}[/]")
        raise typer.Exit(code=1)


model_app = typer.Typer(help="Stage 3: the team volume model.", no_args_is_help=True)
app.add_typer(model_app, name="model")


@model_app.command("fit-volume")
def model_fit_volume(
    holdout: Annotated[
        str, typer.Option("--holdout", help="Seasons held out, e.g. 2024,2025.")
    ] = "2024,2025",
    draws: Annotated[int, typer.Option("--draws", help="Monte Carlo draws.")] = 4000,
) -> None:
    """Fit Layers 1 and 2 and report held-out calibration.

    Calibration, not accuracy, is the number to read. These layers barely beat
    a constant on accuracy -- team play counts are close to unpredictable at
    the game level -- and their contribution downstream is the width of the
    distribution, not the position of its centre.
    """
    import polars as pl

    from nfl_usage_props.features.dataset import build_dataset
    from nfl_usage_props.model.calibration import assess
    from nfl_usage_props.model.team_volume import (
        PLAYS_FEATURES,
        TeamVolumeModel,
        split_by_season,
        team_game_frame,
    )

    config = load_config()
    _, features = build_dataset(config)
    team_games = team_game_frame(features).filter(pl.col("season_type") == "REG")

    holdout_seasons = _parse_seasons(holdout) or []
    train, test = split_by_season(team_games, holdout_seasons)
    console.print(f"[bold]train[/] {train.height:,} team-games   [bold]test[/] {test.height:,}")

    model = TeamVolumeModel(seed=config.model.random_seed)
    fit = model.fit(train)

    for name, result in (("plays (NB)", fit.plays), ("pass rate (Beta-Bin)", fit.pass_rate)):
        table = Table(title=f"Layer: {name}", show_edge=False)
        table.add_column("term")
        table.add_column("coefficient", justify="right")
        for term, coefficient in result.describe():
            table.add_row(term, f"{coefficient:+.4f}")
        table.add_row("[dim]dispersion[/]", f"[dim]{result.dispersion:.1f}[/]")
        console.print(table)
        if not result.converged:
            console.print(f"  [yellow]did not converge cleanly: {result.message}[/]")

    if fit.wind_coefficient is not None:
        console.print(
            f"\n[dim]wind: {fit.wind_coefficient:+.5f} log-odds of passing per mph "
            "(fitted only — observed wind is not available pre-kickoff)[/]"
        )

    usable = model._usable(test, PLAYS_FEATURES)
    sampled = model.sample(test, draws=draws)
    console.print()
    for key, actual in (("plays", "team_plays"), ("dropbacks", "team_dropbacks")):
        report = assess(sampled[key], usable[actual].to_numpy())
        colour = "green" if report.well_calibrated else "yellow"
        console.print(f"[bold]{key:10s}[/] [{colour}]{report.summary()}[/]")


@model_app.command("project")
def model_project(
    holdout: Annotated[
        str, typer.Option("--holdout", help="Seasons held out and projected.")
    ] = "2024,2025",
    draws: Annotated[int, typer.Option("--draws", help="Monte Carlo draws.")] = 4000,
    out: Annotated[str | None, typer.Option("--out", help="Write the PMF table here.")] = None,
) -> None:
    """Fit every layer, project the held-out seasons and report calibration.

    Read the PIT deviation first. Accuracy on these markets is limited by how
    unpredictable football is; whether the distribution is honest about its own
    uncertainty is the part that decides whether an edge is real.
    """
    import polars as pl

    from nfl_usage_props.features.dataset import build_dataset
    from nfl_usage_props.model.calibration import assess
    from nfl_usage_props.model.player_share import (
        PlayerShareModel,
        fit_catch_rate_dispersion,
    )
    from nfl_usage_props.model.projection import project_week
    from nfl_usage_props.model.team_volume import (
        TeamVolumeModel,
        split_by_season,
        team_game_frame,
    )

    config = load_config()
    _, features = build_dataset(config)
    features = features.filter(pl.col("season_type") == "REG")
    team_games = team_game_frame(features).filter(pl.col("season_type") == "REG")

    holdout_seasons = _parse_seasons(holdout) or []
    train_features, test_features = split_by_season(features, holdout_seasons)
    train_teams, test_teams = split_by_season(team_games, holdout_seasons)

    volume = TeamVolumeModel(seed=config.model.random_seed)
    volume.fit(train_teams)
    targets = PlayerShareModel("targets", seed=config.model.random_seed)
    carries = PlayerShareModel("carries", seed=config.model.random_seed)
    target_fit = targets.fit(train_features)
    carry_fit = carries.fit(train_features)
    catch_fit = fit_catch_rate_dispersion(train_features)

    console.print(f"[bold]Layer 3[/] {target_fit.summary()}")
    console.print(f"[bold]Layer 3[/] {carry_fit.summary()}")
    console.print(f"[bold]Layer 4[/] {catch_fit.summary()}")

    result = project_week(
        test_features,
        test_teams,
        volume,
        targets,
        carries,
        catch_fit.concentration,
        draws=draws,
        seed=config.model.random_seed,
    )
    console.print(f"\nprojected [bold]{result.keys.height:,}[/] player-games\n")

    actual = test_features.select("game_id", "team", "gsis_id", "receptions", "carries")
    joined = result.keys.with_row_index("_i").join(
        actual, on=["game_id", "team", "gsis_id"], how="inner"
    )
    index = joined["_i"].to_numpy()
    for market, draws_array in (
        ("receptions", result.receptions),
        ("carries", result.carries),
    ):
        report = assess(draws_array[index], joined[market].to_numpy())
        colour = "green" if report.well_calibrated else "yellow"
        console.print(f"[bold]{market:11s}[/] [{colour}]{report.summary()}[/]")

    if out:
        from pathlib import Path

        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.pmf_table("receptions").write_parquet(path)
        console.print(f"\n[green]wrote receptions PMF[/] -> {path}")
