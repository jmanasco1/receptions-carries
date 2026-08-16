# Test fixtures

## PROVENANCE — READ THIS

`event_odds.json`, `events.json` and `game_odds.json` were **hand-constructed
from The Odds API v4 documentation**, not captured from a live response.

The build environment for Stage 1 had no outbound access to
`api.the-odds-api.com` (blocked by the network egress proxy) and no API key, so
the "make exactly one real call to verify the schema" step could not be
performed. See the README section "What could not be verified".

These fixtures are therefore a **specification of the expected shape**, not
evidence of it. The parser tests prove the client handles that shape; they do
not prove the shape is right.

To replace them with real data:

    nfl-props odds verify-markets --save-fixture

That writes `event_odds_live.json`. Once you have it, diff it against
`event_odds.json` and update the latter (plus any test that depends on it).

`event_odds_live.json` is gitignored-by-convention only in the sense that it is
your captured data — commit it if you want the repo to carry a verified
fixture, which is the recommended end state.
