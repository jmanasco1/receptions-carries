"""Stage 2: the player-game panel and the as-of feature matrix built from it.

Two things live here and they must not be confused:

* `panel` is *what happened* -- one row per player-game of realised usage.
  Every column in it is post-game truth. It is the training target and the
  history, never a same-game feature.
* `build` is *what was knowable before kickoff*. Every column it emits is
  derived from strictly earlier games plus genuinely pre-kickoff information.

`leakage` enforces the boundary mechanically rather than by convention.
"""

from nfl_usage_props.features.panel import build_panel, player_game_usage, team_game_totals

__all__ = ["build_panel", "player_game_usage", "team_game_totals"]
