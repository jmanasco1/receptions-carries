# nfl-usage-props

Probability distributions for two NFL player prop markets — **receptions**
(`player_receptions`) and **rush attempts** (`player_rush_attempts`) — compared
against sportsbook lines to find positive-EV bets.

These markets are the target because they are usage-driven. A receiving yardage
outcome swings on one broken tackle; a reception outcome swings on whether the
ball was thrown to the guy. Usage is more predictable than efficiency, and books
price usage markets less sharply than yardage markets.

> **Status: Stage 1 of 8 complete.** Data ingestion, storage, the odds client,
> and the data dictionary exist and run. **No model exists yet.** Nothing in
> this repo currently projects anything or prices a bet. See
> [Stage 1 status](#stage-1-status) for exactly what runs and what does not.

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
| 3 | Player share of targets / carries | **Dirichlet-Multinomial** across the roster | Recency-weighted history, depth chart, injuries |
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
- **Recency weighting is a fitted parameter.** `model.share_half_life_games` is
  in `config.toml` with a **placeholder of 6.0**. It has *not* been fitted. The
  4–8 game range is a starting bracket, not an answer.

---

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync --extra dev                    # install
uv run pytest -m "not network"         # 131 tests, offline, ~1s
cp .env.example .env                   # add ODDS_API_KEY when you have one

uv run nfl-props config                # show resolved configuration
uv run nfl-props ingest run            # pull nflverse 2016–present (~55s, 165 MB)
uv run nfl-props ingest status         # what landed on disk
uv run nfl-props docs data-dictionary  # regenerate docs/DATA_DICTIONARY.md
```

To pull a subset while trying things out:

```bash
uv run nfl-props ingest run --seasons 2023-2024
uv run nfl-props ingest run --tables pbp,participation --seasons 2024
```

### Storage layout

```
data/raw/<table>/season=<YYYY>/part.parquet   # nflverse tables
data/odds/<endpoint>/<timestamp>.json         # raw odds snapshots, never overwritten
data/metadata.sqlite                          # run log, credit balances, API errors
```

Writes are atomic (temp file then rename), so an interrupted run never leaves a
truncated parquet that a later run mistakes for a complete season.

### Idempotency

A `(table, season)` pair is re-downloaded only if the parquet is missing **or**
the season is not yet complete. A season becomes complete when every game has a
result *and* `storage.completion_lag_days` (default 30) have passed since the
last game — the lag exists because nflverse back-fills corrections for weeks
after a season ends.

Verified on the full corpus: the second `ingest run` drops from **54s to 3.0s**,
with 80 skipped and 3 fetched — `players` plus the two 2026 tables that are
still accumulating, all of which refresh by design.

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

- **`nfl-props ingest run` against live nflverse, full corpus** — all 9 tables,
  2016–present, from an empty `data/`: 83 fetched, 6 pending, **0 errors**, 54s,
  **165 MB**. Row counts: pbp 484,254 · participation 478,989 · rosters_weekly
  466,283 · snap_counts 253,106 · player_stats 182,255.
- **Idempotency on the full corpus** — second run: 80 skipped, 3 fetched
  (`players` and the two 2026 tables still accumulating, both by design), 3.0s.
- **Future-season handling** — 2026 correctly reports 6 tables `pending` and
  exits 0.
- `nfl-props ingest status`, `nfl-props config`, `nfl-props odds credits`.
- `nfl-props docs data-dictionary` — 729 columns across 9 tables, with
  per-column season coverage.
- **131 tests pass offline** in ~1s, plus 1 network-marked test against live
  nflverse. `ruff check` and `ruff format --check` clean.
- Kickoff conversion — verified exact on 2024: 0 nulls in 285 games, 20:20 ET →
  00:20 UTC next day, and February games correctly use EST not EDT.
- Routes proxy — participation joins pbp pass plays 20,007/20,007 for 2024.

### What is stubbed or mocked

- **The odds client has never made a real call.** Every one of its 30 tests runs
  against `tests/fixtures/*.json`, which were **hand-constructed from The Odds
  API v4 documentation, not captured from a live response**. See
  [`tests/fixtures/README.md`](tests/fixtures/README.md). The tests prove the
  client handles that shape correctly; they do not prove the shape is right.
- **`player_receptions` and `player_rush_attempts` are unverified market keys.**
  They are what the docs describe and what the config uses, but nothing in this
  repo has confirmed them against the live API.
- **No model code exists.** `[model]` in `config.toml` holds placeholders
  (`share_half_life_games = 6.0`, `monte_carlo_draws = 10000`) that are loaded
  and tested but consumed by nothing.

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
uv run pytest -m "not network" -q      # 131 pass, ~1s
uv run pytest -m network -q            # 1 test, hits real nflverse
uv run ruff check . && uv run ruff format --check .

rm -rf data
uv run nfl-props ingest run            # ~55s: 83 fetched, 6 pending, 0 errors
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

### What is blocking Stage 2

Nothing blocks feature engineering — the full corpus is on disk and the timing
classification exists. Two things are worth settling first:

1. **Verify the market keys** (the command above). Building a parser around an
   unverified key is exactly what you warned against, and it is a 2-credit
   question I could not answer from here.
2. **Decide the `depth_charts` policy**, which is now two decisions rather than
   one. For 2025+ the `dt` timestamp makes it exact, so there is nothing to
   decide. For ≤2024 the lag rule is still a judgement call — see
   [Open questions](#open-questions). You may also reasonably decide to train
   only on 2025+ depth chart data and skip the legacy feed entirely, at the cost
   of most of the corpus.

---

## Open questions

Three things where the right call depends on information I do not have. I have
not buried a default for any of them.

1. **Do you have a paid Odds API plan, or is 500 credits/month the real
   constraint?** It changes the whole ingestion cadence and decides whether
   historical prop backtesting is on the table at all. The config assumes free
   tier.
2. **`depth_charts` lag rule for ≤2024 only** — accept the leak and use week-N
   charts, or lag to week N-1 and lose Wednesday role changes? I lean N-1, and
   would make it a config flag so the leakage test can assert against it either
   way. 2025+ needs no rule: `dt` makes it exact. A third option is to use only
   2025+ depth charts and drop the legacy feed, which is clean but throws away
   nine seasons of depth-chart signal.
3. **Scope: which players get projected?** Everyone with a prop line posted, or
   everyone above a usage floor regardless of whether a line exists? The second
   is more work but is what lets you notice a book has *not* posted a line on
   someone who should have one.

---

## Layout

```
config.toml                     all tuning, no secrets
.env.example                    ODDS_API_KEY lives in .env (gitignored)
src/nfl_usage_props/
  config.py                     TOML + env loading
  storage.py                    season-partitioned parquet, atomic writes
  metadata.py                   SQLite run log, credit balances, API errors
  datadict.py                   timing classification + doc generation
  cli.py                        nfl-props
  ingest/nflverse.py            table registry, completeness, idempotency
  odds/client.py                Odds API client, credit guards, retries
docs/DATA_DICTIONARY.md         generated — 729 columns
tests/                          131 offline tests + 1 network-marked
.github/workflows/ci.yml        lint + offline tests; Stage 2's leakage test lands here
```

## Data sources

- **[nflreadpy](https://github.com/nflverse/nflreadpy)** ≥ 0.1.5 — the current
  nflverse Python package. Verified at build time: `nfl_data_py` is deprecated
  in favour of it, and all future nflverse Python development is here.
- **[The Odds API](https://the-odds-api.com/)** v4 — spreads, totals, props.

## Roadmap

1. ✅ Scaffold + data ingestion
2. Feature engineering with as-of joins and the leakage test
3. Team volume model (plays, pass/rush split)
4. Player share models (Dirichlet-Multinomial, Beta-Binomial)
5. Monte Carlo composition → PMFs
6. Calibration harness (reliability diagrams, log loss by decile)
7. Odds ingestion, devig, edge calculation, fractional Kelly sizing
8. GitHub Actions automation + weekly report output
