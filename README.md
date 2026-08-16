# nfl-usage-props

Probability distributions for two NFL player prop markets — **receptions**
(`player_receptions`) and **rush attempts** (`player_rush_attempts`) — compared
against sportsbook lines to find positive-EV bets.

These markets are the target because they are usage-driven. A receiving yardage
outcome swings on one broken tackle; a reception outcome swings on whether the
ball was thrown to the guy. Usage is more predictable than efficiency, and books
price usage markets less sharply than yardage markets.

> **Status: Stage 1 of 8 complete.** Data ingestion, player identity resolution,
> prop snapshot logging, the odds client and the data dictionary exist and run.
> **No model exists yet.** Nothing in this repo currently projects anything or
> prices a bet. See [Stage 1 status](#stage-1-status) for exactly what runs and
> what does not.
>
> **If you do one thing first, start the snapshot logger.** Closing line value
> is the only validation signal available on the free tier and it cannot be
> reconstructed later — every unlogged week is gone permanently, model or no
> model.

---

## The architecture this is being built toward

Player reception and carry counts are **not** modelled directly as a function of
features. That flattens a hierarchical process into a regression and throws away
the constraints that make it accurate. The generative chain is modelled instead:

```
Game environment  →  Team volume  →  Pass/rush split  →  Player share  →  Conversion
   (spread, total)      (plays)         (PROE, script)     (Dirichlet)     (catch rate)
```

| Layer | Quantity | Method | Driven by |
| --- | --- | --- | --- |
| 1 | Team offensive plays | Negative Binomial | Neutral-pace of both teams, game total, expected ToP |
| 2 | Pass / rush split | Beta-Binomial on plays | Spread (game script), team PROE, opponent tendency |
| 3 | Player share of targets / carries | **Dirichlet-Multinomial**, partially pooled | State-space role tracker, depth chart, injuries |
| 4 | Targets → receptions | **Beta-Binomial** | Catch-rate prior by position and depth role |

Composition is by **Monte Carlo** — 10,000 draws per player per game, PMF read
off the samples. The compound distribution is sampled, not approximated.

**Why this beats a flat GLM.** Injury news enters cleanly at Layer 3: a WR1
ruled out means redistributing Dirichlet concentration, not encoding
"teammate absent" as a feature. Market information enters at Layers 1–2 where it
belongs — Vegas knows the game environment, Vegas does not know the target
share. And every layer is independently testable and calibratable.

### Design commitments

- **PMFs, not means.** A mean of 4.2 receptions says nothing about a 3.5 line
  versus a 4.5 line. The half-point is the entire bet. The output is
  `P(X >= 4)` and `P(X >= 5)`, never `E[X]`.
- **Overdispersion is assumed real.** Count layers use Negative Binomial and the
  dispersion parameter gets fitted and logged, not assumed to be Poisson.
- **Shrinkage everywhere.** Empirical-Bayes Beta-Binomial priors from the
  population of same-position, same-depth-rank players. A rookie WR3 with 4
  targets in 2 games must not project a 67% target share.
- **Divergence from the market is not edge.** Usage changes are discontinuous —
  a WR1 goes on IR and the WR3's target share steps from 11% to 22% in one week.
  Exponential decay assumes gradual change and would still project ~15% three
  weeks later, while the book repriced after one game. The model would then read
  its own staleness as edge, and it would do so precisely when the situation is
  most interesting. So there is **no fixed decay half-life anywhere in this
  project**; see the role tracker below.

### The role tracker (replaces fixed decay entirely)

Latent role as a random walk, weekly observation with noise scaled by exposure,
Kalman-style update. This handles byes and unequal gaps natively and returns
uncertainty rather than a point estimate.

- **Per-layer speed.** Snap share settles fastest (~4 games), carry share ~5–6,
  target share slowest (~7–8). One global rate is wrong for most of them, so
  process noise is configured per layer.
- **Uncertainty propagates into the Monte Carlo** as parameter uncertainty.
  Collapsing to a point estimate and sampling around it understates variance for
  exactly the players whose roles just changed.
- **Process noise spikes on known role-change events** — teammate to IR, trade,
  coordinator change, depth-chart promotion, first game back. Fast when reality
  jumped, slow otherwise.
- **Observations weight by exposure, not just recency.** A 38-route game is
  stronger evidence than a 12-route game regardless of order, and games below a
  configurable share of the player's trailing median exposure are treated as
  partial (injury exit, blowout benching) and downweighted.
- **Changepoint suppression.** When recent games diverge from the trailing
  estimate beyond a threshold, the projection is marked low-confidence and
  suppressed. Skipping beats acting on stale.

Catch rate is a separate mechanism: a Beta prior strength of ~50 targets, not a
decay rate.

---

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync --extra dev                    # install
uv run pytest -m "not network"         # 222 tests, offline, ~1s
cp .env.example .env                   # add ODDS_API_KEY when you have one

uv run nfl-props config                # show resolved configuration
uv run nfl-props ingest run            # pull nflverse 2016–present (~56s, 165 MB)
uv run nfl-props ingest status         # what landed on disk
uv run nfl-props docs data-dictionary  # regenerate docs/DATA_DICTIONARY.md
```

Once you have an API key, the time-critical part:

```bash
uv run nfl-props odds verify-markets   # ~2 credits, confirms the market keys
uv run nfl-props odds snapshot --dry-run   # free: what would be pulled, and cost
uv run nfl-props odds snapshot --kind open # ~32 credits, starts the CLV record
uv run nfl-props odds resolve          # map book names to gsis_ids
uv run nfl-props odds credits          # balance, no API call
```

To pull a subset while trying things out:

```bash
uv run nfl-props ingest run --seasons 2023-2024
uv run nfl-props ingest run --tables pbp,participation --seasons 2024
```

### Storage layout

```
data/raw/<table>/season=<YYYY>/part.parquet        # nflverse tables (gitignored)
data/odds/props/season=<YYYY>/<id>.parquet         # parsed prop snapshots (VERSIONED)
data/odds/<endpoint>/<timestamp>.json              # raw payloads, never overwritten
data/metadata.sqlite                               # run log, credits, API errors
```

Everything under `data/` is gitignored **except `data/odds/props/`**. nflverse
data is reproducible with one command; prop snapshots are not reproducible at
any price on this tier, so they are versioned rather than left to die with a
container.

Writes are atomic (temp file then rename), so an interrupted run never leaves a
truncated parquet that a later run mistakes for a complete season.

### Idempotency

A `(table, season)` pair is re-downloaded only if the parquet is missing **or**
the season is not yet complete. A season becomes complete when every game has a
result *and* `storage.completion_lag_days` (default 30) have passed since the
last game — the lag exists because nflverse back-fills corrections for weeks
after a season ends.

Verified on the full corpus: the second `ingest run` drops from **56s to 3.0s**,
with 80 skipped and 4 fetched — `players` and `teams` (current-state snapshots)
plus the two 2026 tables that are still accumulating, all of which refresh by
design.

### Future seasons are `pending`, not errors

Run this in August and the upcoming season is on the schedule but has not been
played. nflverse publishes `schedules` and `depth_charts` for it months ahead;
`pbp`, `snap_counts`, `participation` and the rest simply do not exist yet.

Those are reported as **`pending`** and do not fail the run or set a non-zero
exit code — a season that has not happened is not a failure. The distinction is
deliberately narrow, so a genuine mid-season outage still surfaces as an error:

- `ValueError: Season must be between X and Y` is upstream declaring what it
  has → always `pending`.
- A download failure is `pending` **only** when no game of that season has been
  played yet, judged from the schedule. During a live season the same failure is
  a real error.

---

## Credit budget — this constrains the whole project

The Odds API free tier is **500 credits/month**.

| Pull | Endpoint | Cost |
| --- | --- | --- |
| Spreads + totals, whole slate | `/v4/sports/americanfootball_nfl/odds` | 2 credits |
| Player props, **per event** | `/v4/sports/.../events/{eventId}/odds` | markets × regions |
| Event list | `/v4/sports/.../events` | free |

A full-slate props pull is roughly **16 events × 2 markets × 1 region = 32
credits**. Two pulls a week (Wednesday open, Sunday morning close) across four
game weeks is **~256 credits/month**, leaving almost no margin for retries,
failed runs, or a third snapshot.

### The snapshot schedule

Closing lines are captured **per game day**, not in one Sunday pull. A Thursday
game closes Thursday; a Sunday-morning pull would capture the TNF line after the
game had already been played.

| When (UTC) | What | Credits |
| --- | --- | --- |
| Wed 18:00 | opener, whole week | ~32 |
| Thu 23:00 | TNF close | 2 |
| Sun 16:00 | close, Sunday slate | ~26 |
| Sun 19:30 | close, late window | ~8 |
| Sun 23:30 | SNF close | 2 |
| Mon 23:00 | MNF close | 2 |
| | **~72/week, ~310/month** | |

That leaves ~190 credits/month of the free tier's 500 for retries and re-polls.
Runs that fire with nothing in window cost one free `/events` call and exit,
because event selection skips anything within `close_cutoff_minutes` of kickoff.

`.github/workflows/snapshot.yml` runs this and commits the results. **Prop
snapshots are the one thing under `data/` that is versioned in git** — they
cannot be re-downloaded at any price on this tier, so they are not left to die
with a container.

### The re-poll compromise

4B's stale-price filter wants an outlier confirmed across two consecutive polls,
but two polls a week means a Sunday outlier has no confirmation before kickoff —
the filter could never fire when it matters. A blanket third poll costs ~138
credits/month and does not fit. So re-polls are **targeted**: only events
carrying a flagged outlier are pulled again (`odds.snapshots.max_repoll_events`,
a handful of events at ~4–8 credits). Cheap, and it confirms exactly the prices
the filter is about to act on.

Credit accounting is therefore built in from Stage 1, not bolted on:

- `x-requests-remaining`, `x-requests-used` and `x-requests-last` are parsed off
  every response and written to `data/metadata.sqlite`.
- The client **hard-aborts before a call** when the balance would drop below
  `odds.credits.floor` (default 50). The refused call never reaches the network.
- A fresh process reads the last known balance out of SQLite, so restarting does
  not reset the guard.
- Bulk props pulls estimate total cost up front and refuse to start if it
  exceeds `odds.credits.max_spend_per_run` — a slate that unexpectedly has 40
  events cannot silently drain the quota.
- 429 and 5xx retry with exponential backoff. **401/404/422 do not retry** — a
  wrong market key will never succeed, and retrying it just burns credits.

Check your balance without spending anything:

```bash
uv run nfl-props odds credits
```

---

## Things that will not work

Stated plainly, because each one changes what conclusions this repo can support.

### You cannot backtest ROI

The Odds API's historical props go back to 2023-05-03 but are **paid-plan only
and cost 10× credits**. Without historical lines you can validate the
*probability model* — log loss, Brier score, calibration curves against actual
game outcomes — but you **cannot measure edge or CLV historically**.

A calibration harness (Stage 6) will tell you whether the model's stated 62% is
really 62%. It will not tell you whether that was a profitable bet. Any number
in this repo that looks like historical ROI is not one, and the backtest harness
must never imply otherwise.

### There is no sharp book

The Odds API carries ~40 mainstream soft books. **No Pinnacle, no Betfair
Exchange.** There is no sharp anchor to calibrate against, so "market consensus"
here means a consensus of soft books — a materially weaker reference than a
sharp line. Devigging a soft consensus gives you a soft consensus, not truth.

### Prop limits are small

Model precision matters less than line shopping. Output is therefore designed to
show the **best available number across books**, not a single-book edge. A 4%
edge you can bet $50 into at the best price beats a 6% edge at a book that will
book you $12.

### No bet sizing, deliberately

There is no Kelly, fractional Kelly, bankroll tracking or stake recommendation
anywhere in this project, and none is planned in any stage. The output is
probabilities, fair lines, and where the number sits versus the market. What to
do with that is out of scope.

### Book names do not match nflverse ids

The Odds API keys props by book-written name strings; nflverse keys on
`gsis_id`. A resolver that quietly drops what it cannot match looks like it
works while losing part of the slate — and the losses are not random. They
concentrate in rookies, suffixed names and just-traded players, which is
disproportionately where role change, and therefore edge, lives.

`nfl-props odds resolve` **never drops**. An unmappable name raises, and the
error message is directly pasteable into
`reference/player_name_overrides.csv`.

Two findings from running it against the real 25k-player nflverse universe,
neither of which the hand-built fixture could have surfaced:

- **Father/son pairs are real and suffix-stripping collapses them.** Marvin
  Harrison / Marvin Harrison Jr. and Michael Pittman / Michael Pittman Jr. are
  distinct players. Worse, nflverse stores the Pittmans with *no* suffix on
  either, so the suffix cannot always break the tie.
- **142 duplicate skill-player names exist.** Recency separates 137 of them
  (books only price active players); the 5 that remain tied are all retired and
  a test asserts they stay that way — an active tie would need an override row.

Resolution order: manual override → exact normalized → first-initial + surname →
fuzzy above threshold, and only when unambiguously best. Candidates are scoped
to the two teams in the event first, which resolves the "Josh Allen is both a QB
and an edge rusher" collision structurally rather than statistically, with a
full-universe fallback so mid-season trades still resolve.

### Wind only

Wind is the only weather variable used. Above ~15mph pass attempts fall and rush
attempts rise; temperature and precipitation are deliberately ignored because
their apparent effects are confounded with team quality and game total.

`schedules.roof` records the **observed** state on the day. For five retractable
stadiums that is a game-time decision, so the column is classified `after` and
using it as a forward feature would leak. `reference/stadiums.csv` carries the
fixed `roof_type` (plus coordinates) keyed on `stadium_id` — which survives
renames, unlike the stadium name. `reference.is_wind_shielded` keeps the two
apart, and treats an unknown retractable state as *not* shielded: modelling wind
that turned out to be absent is a smaller error than zeroing out wind that was
really blowing.

`schedules.wind` is null for 325 of 855 games — structurally zero for domes
rather than missing, and the two must not be conflated.

### Routes run is the correct denominator, and we do not have it

**Receptions should be modelled per route run, not per snap.** A blocking TE and
a slot WR at identical snap shares have completely different target opportunity.
Routes-run data is PFF's and is paywalled — it is not in nflverse core.

What was actually checked, rather than assumed:

- **FTN charting (`load_ftn_charting`) does not contain route data.** Verified
  against the real 2024 table: it is play-level charting (`is_play_action`,
  `n_pass_rushers`, `is_screen_pass`, …) with no per-player route
  participation. The prompt's suggestion to check it was worth checking; the
  answer is no.
- **`load_participation` is better than snap share.** Its `offense_players`
  column lists the 11 gsis_ids on the field for every play. Filtered to pass
  plays this gives **pass-play participation**, available **2016–2025**, which
  joins to pbp pass plays **20,007 / 20,007 (100%)** for 2024. Target-per-pass-
  snap for high-volume WRs lands at 0.24–0.32, a plausible range.
- **It is still a proxy, not routes run.** A TE who stays in to block is on the
  field for a dropback and counts as a pass snap while running no route. It
  removes run-play snaps from the denominator, which is the larger of the two
  errors, but it does not remove pass-protection snaps.

The plan is to use pass-play participation as the Layer 3 exposure term and
carry the residual bias knowingly. `snap_counts.offense_pct` remains ingested as
a fallback and cross-check.

---

## Leakage

Every feature for a week-N game must be computable from data timestamped before
that game's kickoff. `schedules.kickoff_utc` (derived on ingest: `gameday` +
`gametime` parsed as US/Eastern, converted to UTC) is the cutoff for all of it.

The generated [data dictionary](docs/DATA_DICTIONARY.md) classifies all **729
ingested columns** as `before` / `after` / `static` / `lagged`, and records which
seasons each column actually appears in. Three traps are worth stating here:

1. **`rosters_weekly` and pre-2025 `depth_charts` are `lagged`, not `before`.**
   They carry a `week` but no timestamp, and nflverse overwrites one snapshot
   per week. A row labelled week 5 may reflect a chart published *after* the
   week 5 game. Using week-N data for the week-N game is leakage that no schema
   check catches. Use week N-1, or accept the leak knowingly.
2. **`schedules.temp` and `schedules.wind` are observed, not forecast** — they
   are `after`, despite sitting in an otherwise pre-game table. Same for
   `away_qb_id`/`home_qb_id`, which record who actually started.
3. **`depth_charts` changed schema completely in 2025** — see below. It is the
   one case where the leakage rule differs by season.

### depth_charts has two incompatible schemas

Found by ingesting the full corpus, not by reading docs:

| | ≤ 2024 | ≥ 2025 |
| --- | --- | --- |
| Key columns | `season`, `week`, `club_code`, `depth_team` | `dt`, `team`, `pos_rank`, `pos_slot` |
| Timestamp | none | `dt`, ISO-8601 UTC |
| Snapshots | one per week, overwritten | 221 distinct in 2025 |
| Rows (season) | 37,312 (2024) | 554,215 (2025) |
| Timing class | `lagged` | `before` |

The 2025+ feed is **exactly as-of filterable** against `kickoff_utc`, so no lag
heuristic is needed there — but it has **no `week` column at all**, so it must be
joined on time rather than week. Any Stage 2 code touching depth charts has to
handle both shapes; the dictionary marks every column with the seasons it
appears in so this cannot be missed.

**The leakage test itself is Stage 2 and does not exist yet.** The timing
metadata it will consume exists and is unit-tested; the test that samples 200
historical rows and asserts reproducibility from a truncated dataset has not
been written. Until it exists, treat any feature-level claim as unverified.

---

## Stage 1 status

### What runs end to end, verified by running it

- **`nfl-props ingest run` against live nflverse, full corpus** — all 10 tables,
  2016–present, from an empty `data/`: 84 fetched, 6 pending, **0 errors**, 56s,
  **165 MB**. Row counts: pbp 484,254 · participation 478,989 · rosters_weekly
  466,283 · snap_counts 253,106 · player_stats 182,255.
- **Idempotency on the full corpus** — second run: 80 skipped, 4 fetched
  (`players`, `teams` and the two 2026 tables still accumulating, all by
  design), 3.0s.
- **Future-season handling** — 2026 correctly reports 6 tables `pending` and
  exits 0.
- `nfl-props ingest status`, `nfl-props config`, `nfl-props odds credits`,
  `nfl-props odds snapshot --dry-run`, `nfl-props odds resolve`.
- `nfl-props docs data-dictionary` — 745 columns across 10 tables, with
  per-column season coverage.
- **222 tests pass offline** in ~1s, plus **30 network-marked** against live
  nflverse. `ruff check` and `ruff format --check` clean.
- Kickoff conversion — verified exact on 2024: 0 nulls in 285 games, 20:20 ET →
  00:20 UTC next day, and February games correctly use EST not EDT.
- Routes proxy — participation joins pbp pass plays 20,007/20,007 for 2024.
- **Player resolution against the real 25k-player universe** — suffix variants,
  punctuation variants, father/son pairs, traded players and same-name
  collisions all covered by tests that run against ingested nflverse data, not
  against invented names.
- **Full pipeline integration** — fixture payload → parse → hold computation →
  snapshot store → resolve, with every name resolving to a distinct `gsis_id`.
- Stadium roof types **derived from 10 seasons of observed data**, not from
  memory: 5 retractable, 6 dome, the rest outdoor.

### What is stubbed or mocked

- **The odds client has never made a real call.** Every one of its 30 tests runs
  against `tests/fixtures/*.json`, which were **hand-constructed from The Odds
  API v4 documentation, not captured from a live response**. See
  [`tests/fixtures/README.md`](tests/fixtures/README.md). The tests prove the
  client handles that shape correctly; they do not prove the shape is right.
- **`player_receptions` and `player_rush_attempts` are unverified market keys.**
  They are what the docs describe and what the config uses, but nothing in this
  repo has confirmed them against the live API.
- **No model code exists.** `[model]`, `[model.role_tracker]`,
  `[model.catch_rate]`, `[model.early_season]` and `[edge]` in `config.toml`
  hold parameters that are loaded and tested but consumed by nothing. The
  process-noise values are chosen to match observed stabilisation speeds; they
  are **priors and sanity bounds, not fitted values**.
- **The snapshot workflow has never fired.** `.github/workflows/snapshot.yml`
  is written and its cron schedule is costed, but it needs an `ODDS_API_KEY`
  repository secret and has not run once. Its cadence assumes US Eastern
  daylight kickoff times.
- **The weather forecast client is not built.** Deferred to Stage 3 on purpose:
  unlike CLV, weather has no retroactive-loss property — observed wind is
  already in nflverse for fitting, and forecasts are only needed once
  projections exist. `reference/stadiums.csv` (coordinates + roof type) is
  committed now because the roof-leakage fix needs it regardless.

### What could not be verified

- **The Odds API is unreachable from this build environment.**
  `api.the-odds-api.com` is blocked by the network egress proxy, and no
  `ODDS_API_KEY` was available. The prompt's "make exactly one real call to
  verify the schema and capture a fixture" step **was not performed** — not
  skipped by choice, but impossible here. This is the single biggest gap in
  Stage 1.
- Consequently unverified: the market keys, the exact credit cost of an event
  odds call, whether `x-requests-last` is the header name for per-call cost, and
  the real response shape.
- **How books actually spell names.** The resolver is tested hard against the
  real nflverse universe, but the *input* side is still hypothetical — I do not
  know whether the feed writes "Marvin Harrison Jr.", "M. Harrison" or something
  else, nor which books appear in the `us` region. Expect
  `reference/player_name_overrides.csv` to need real entries in week 1.
- **Whether Circa or BetOnline are in this feed at all.** The consensus weights
  in `[edge]` name them, but that is from the spec, not from a response I have
  seen. If they are absent the weights are inert and consensus is an unweighted
  soft-book average.
- **Open-Meteo is also blocked** by this environment's proxy (403), so even the
  deferred weather client could not have been verified here.
- **Data quality beyond schema and row counts.** The full 2016–present corpus
  ingests cleanly, but I have not audited the *values* — e.g. how often
  `receiver_player_id` is null on a play that was clearly a target, or whether
  snap counts reconcile against participation. That belongs in Stage 2.
- **2016–2022 participation has ~9% of rows with an empty `offense_players`**
  (2016: 44,341 of 48,434 populated). 2023+ is 100%. I have not established
  whether those gaps are random or concentrated in particular games, which
  matters for the exposure term.

### Confirm it yourself

```bash
uv sync --extra dev
uv run pytest -m "not network" -q      # 222 pass, ~1s
uv run pytest -m network -q            # 30 tests, hits real nflverse
uv run ruff check . && uv run ruff format --check .

rm -rf data
uv run nfl-props ingest run            # ~56s: 84 fetched, 6 pending, 0 errors
uv run nfl-props ingest run            # ~3s:  80 skipped        <- idempotency
uv run nfl-props ingest status
uv run nfl-props docs data-dictionary && git diff --stat docs/   # should be empty
```

Then, with a real key in `.env` — **this is the one call I could not make**:

```bash
uv run nfl-props odds verify-markets --save-fixture
```

It makes one free `/events` call plus **one** event-odds call (~2 credits),
prints `FOUND` / `MISSING` per configured market key, writes a snapshot to
`data/odds/verify/`, and saves `tests/fixtures/event_odds_live.json`. Diff that
against `tests/fixtures/event_odds.json` and update the hand-built fixture. It
exits non-zero if either market key is missing.

Then, to start the CLV record — **this is the time-critical one**:

```bash
uv run nfl-props odds snapshot --dry-run   # free: lists events and cost
uv run nfl-props odds snapshot --kind open # ~32 credits
uv run nfl-props odds resolve              # reports unresolvable names
```

Add `ODDS_API_KEY` as a repository secret to let
`.github/workflows/snapshot.yml` run the schedule automatically. Every week it
does not run is a week of closing lines that cannot be recovered later.

### What is blocking Stage 2

Nothing blocks feature engineering — the full corpus is on disk and the timing
classification exists. Two things are worth doing first, and one of them is on a
clock:

1. **Start the snapshot logger.** Add the `ODDS_API_KEY` secret and let the
   workflow run. This is the only item here with a deadline: every week before
   it starts is a week of closing lines that no amount of later work recovers.
   It does not depend on the model, the resolver, or anything else in Stage 2.
2. **Verify the market keys** (`odds verify-markets`). Building a parser around
   an unverified key is exactly what you warned against, and it is a 2-credit
   question I could not answer from here. Doing this before (1) is sensible —
   a wrong market key would make the first snapshots empty.

The depth-chart lag and report scope are settled and recorded in `config.toml` —
see [Decisions](#decisions).

---

## Decisions

### Settled

**Depth-chart lag: week N-1 for the legacy feed.** `model.depth_chart_lag_weeks
= 1` in `config.toml`. Applies only to the ≤2024 feed, which is an undated
weekly snapshot; the 2025+ feed carries `dt` and is filtered exactly against
`kickoff_utc` instead. Stage 2's leakage test asserts both paths. Setting the
parameter to 0 knowingly accepts the leak, and the leakage test is expected to
fail at 0 for legacy seasons — that is what the parameter is for.

**Report scope: players with a posted line.** `output.scope =
"posted_line_only"`.

One important consequence, because it is easy to get wrong: **this filters the
report, not the estimation.** Layer 3 is a Dirichlet-Multinomial whose shares
must sum to one, so every rostered player is projected internally regardless.
Dropping teammates from the simplex would redistribute their share across the
remainder and bias every surviving projection upward — the model would think a
team throws 100% of its targets to the four players a book happened to price.
Full roster in, filtered list out.

`output.scope = "all_rostered"` additionally reports players with no posted
line, which is what surfaces a book failing to post a line on someone who should
have one. Available, not the default.

**Free tier, confirmed.** 500 credits/month is the real constraint. Historical
prop backtesting is therefore permanently off the table, which is why the
snapshot logger matters as much as it does — it is the only way this project
ever acquires validation data.

**Closing pulls are per game day**, and the stale-price filter uses **targeted
re-polls** rather than a blanket third poll. Both fall out of the credit budget;
see [The snapshot schedule](#the-snapshot-schedule).

**Weather forecasting is deferred to Stage 3**, since observed wind already
supports fitting and forecasts are only needed once projections exist. The
stadium reference table ships now because the roof-leakage fix depends on it.

### Still open — do not invent answers to these

These three resolve from the snapshot log, which is the argument for starting it
now rather than when the model is ready.

1. **Whether line movement should ever suppress a flagged edge, or only
   annotate it.** The diagnostic gets built; the policy waits. Leaning on it too
   hard just makes the model track the market, which has no edge by
   construction.
2. **The 3% minimum edge threshold.** Informed by typical prop hold (6–10% on
   these markets vs ~4.5% on sides), not fitted. Re-derive from realized CLV
   after ~6 weeks of logging.
3. **Consensus weighting of "sharper" books.** There is no clean anchor without
   Pinnacle, and I have not confirmed that Circa or BetOnline even appear in
   this feed. The right weights are empirical.

---

## Layout

```
config.toml                     all tuning, no secrets
.env.example                    ODDS_API_KEY lives in .env (gitignored)
reference/
  stadiums.csv                  coordinates + fixed roof_type, keyed on stadium_id
  player_name_overrides.csv     manual book-name -> gsis_id; expected to grow
src/nfl_usage_props/
  config.py                     TOML + env loading
  storage.py                    season-partitioned parquet, atomic writes
  metadata.py                   SQLite run log, credit balances, API errors
  reference.py                  reference CSV loaders, wind-shielding logic
  datadict.py                   timing classification + doc generation
  cli.py                        nfl-props
  ingest/nflverse.py            table registry, completeness, idempotency
  identity/resolver.py          book name -> gsis_id, never drops
  odds/client.py                Odds API client, credit guards, retries
  odds/parse.py                 payload -> rows, implied probability, hold
  odds/snapshots.py             CLV logging, per-game-day close selection
data/odds/props/                prop snapshots — the one versioned data dir
docs/DATA_DICTIONARY.md         generated — 745 columns
tests/                          222 offline tests + 30 network-marked
.github/workflows/ci.yml        lint + offline tests; Stage 2's leakage test lands here
.github/workflows/snapshot.yml  scheduled CLV logging, commits results
```

## Data sources

- **[nflreadpy](https://github.com/nflverse/nflreadpy)** ≥ 0.1.5 — the current
  nflverse Python package. Verified at build time: `nfl_data_py` is deprecated
  in favour of it, and all future nflverse Python development is here.
- **[The Odds API](https://the-odds-api.com/)** v4 — spreads, totals, props.

## Roadmap

1. ✅ Scaffold, data ingestion, player identity resolution, prop snapshot logging
2. Feature engineering with as-of joins and the leakage test
3. Team volume model (plays, pass/rush split), opponent pace, wind forecast
4. Player share models (state-space role tracker, partially pooled
   Dirichlet-Multinomial, Beta-Binomial catch rate)
5. Monte Carlo composition → PMFs, with role uncertainty propagated
6. Calibration harness (reliability diagrams, log loss by decile)
7. Devig (power / multiplicative / Shin), consensus fair line, edge calculation
8. Weekly report output

**No bet sizing at any stage.** If a stage appears to need it, that is a signal
to stop and ask, not to add it.

### Opponent adjustment, and where it must not be applied

Opponent effects on *volume* are far weaker than on efficiency, and most public
"defense vs position" adjustments are overfit noise.

- **Layer 1** — use opponent neutral-situation pace and plays allowed. Real and
  stable.
- **Layer 2** — use opponent pass rate faced, *adjusted for game script*. A bad
  team faces more rushing because it is losing; that is endogenous and already
  partly in the spread, so adjusting is required or it is double-counted.
- **Layer 3** — **do not** apply DVP-style per-position volume adjustments. Tiny
  samples, mostly noise, and where public models overfit hardest.
- **Layer 3, one exception** — shadow corner assignments. When an elite CB
  travels with a specific WR, that WR's target share genuinely drops. No free
  data feed exists, so this is a small manual weekly CSV, and the model must run
  correctly with the file empty.
