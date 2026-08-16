# Data dictionary

**Generated** by `nfl-props docs data-dictionary` from the parquet files on
disk. Do not edit by hand — edit `src/nfl_usage_props/datadict.py` and
regenerate.

## Timing classes

This column is the whole point of the document. Stage 2's leakage test
asserts that no feature for a week-N game is built from an `after` column
of that same game.

| class | meaning |
| --- | --- |
| `before` | Known before kickoff — safe as a same-game feature. |
| `after` | Known only after the game — history only, never same-game. |
| `static` | Time-invariant identity/biographical field. |
| `lagged` | Pre-game in principle, but undated — needs an explicit week lag. |

### The `lagged` trap

`rosters_weekly` and the pre-2025 `depth_charts` feed carry a `week` but no
timestamp. nflverse stores one snapshot per week and overwrites it, so a row
labelled week 5 may reflect a chart published *after* the week 5 game. Using
week-N data for the week-N game is therefore leakage that no schema check
will catch. Use week N-1, or accept the leak knowingly and document it.

Two tables escape this:

* `injuries` has `date_modified`, a real UTC timestamp.
* **`depth_charts` from 2025 onward** switched to an entirely different feed
  with a `dt` ISO-8601 UTC timestamp (221 distinct snapshots in 2025). Those
  columns are `before`, not `lagged`, and join on time rather than `week` —
  the modern schema has no `week` column at all.

Both are filtered exactly against `schedules.kickoff_utc`.

### Schema drift

The `seasons` column on each table below records which ingested seasons
actually contain that column. Anything other than `all` means the column
appeared, disappeared, or was replaced partway through the corpus — read it
before building a feature on that column.

## Tables

| table | rows (latest season) | seasons | columns | default timing |
| --- | --- | --- | --- | --- |
| [`depth_charts`](#depth_charts) | 439,615 | 2016–2026 | 26 | `before` |
| [`injuries`](#injuries) | 6,068 | 2016–2025 | 17 | `before` |
| [`participation`](#participation) | 45,184 | 2016–2025 | 26 | `after` |
| [`pbp`](#pbp) | 48,771 | 2016–2025 | 372 | `after` |
| [`player_stats`](#player_stats) | 19,422 | 2016–2025 | 150 | `after` |
| [`players`](#players) | 25,033 | n/a (snapshot) | 39 | `before` |
| [`rosters_weekly`](#rosters_weekly) | 46,849 | 2016–2025 | 36 | `before` |
| [`schedules`](#schedules) | 272 | 2016–2026 | 47 | `before` |
| [`snap_counts`](#snap_counts) | 26,612 | 2016–2025 | 16 | `after` |

### `depth_charts`

Depth chart position and rank. **Two incompatible schemas**: through 2024 an undated weekly snapshot (season/week/club_code/depth_team); from 2025 a different feed keyed on a `dt` UTC timestamp (dt/team/pos_rank/pos_slot) with no `week` column. The 2025+ feed is exactly as-of filterable; the legacy one needs a week lag. Any code touching this table must handle both.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `season` | `Int32` | `lagged` | 2016–2024 |  |
| `club_code` | `String` | `lagged` | 2016–2024 |  |
| `week` | `Int32` | `lagged` | 2016–2024 |  |
| `game_type` | `String` | `lagged` | 2016–2024 |  |
| `depth_team` | `String` | `lagged` | 2016–2024 | LEGACY (<=2024) depth rank, 1 = starter. Prior for shrinkage (Layer 3/4). Undated weekly snapshot — needs a week lag. |
| `last_name` | `String` | `lagged` | 2016–2024 |  |
| `first_name` | `String` | `lagged` | 2016–2024 |  |
| `football_name` | `String` | `lagged` | 2016–2024 |  |
| `formation` | `String` | `lagged` | 2016–2024 |  |
| `gsis_id` | `String` | `before` | all |  |
| `jersey_number` | `String` | `lagged` | 2016–2024 |  |
| `position` | `String` | `lagged` | 2016–2024 |  |
| `elias_id` | `String` | `lagged` | 2016–2024 |  |
| `depth_position` | `String` | `lagged` | 2016–2024 |  |
| `full_name` | `String` | `lagged` | 2016–2024 |  |
| `dt` | `String` | `before` | 2025–2026 | MODERN (>=2025) ISO-8601 UTC snapshot timestamp. Makes depth charts exactly as-of filterable against schedules.kickoff_utc — no lag heuristic needed. Note there is no `week` column in this schema; join on time, not week. |
| `team` | `String` | `before` | 2025–2026 |  |
| `player_name` | `String` | `before` | 2025–2026 |  |
| `espn_id` | `String` | `before` | 2025–2026 |  |
| `pos_grp_id` | `String` | `before` | 2025–2026 |  |
| `pos_grp` | `String` | `before` | 2025–2026 |  |
| `pos_id` | `String` | `before` | 2025–2026 |  |
| `pos_name` | `String` | `before` | 2025–2026 |  |
| `pos_abb` | `String` | `before` | 2025–2026 |  |
| `pos_slot` | `Int32` | `before` | 2025–2026 | MODERN (>=2025) ordering slot within the unit. |
| `pos_rank` | `Int32` | `before` | 2025–2026 | MODERN (>=2025) depth rank within position, 1 = starter. |

### `injuries`

Weekly injury report with practice and game status. Carries a real `date_modified` UTC timestamp -- the only table that supports exact as-of filtering without a lag heuristic.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `season` | `Int32` | `before` | all |  |
| `game_type` | `String` | `before` | all |  |
| `team` | `String` | `before` | all |  |
| `week` | `Int32` | `before` | all |  |
| `gsis_id` | `String` | `before` | all |  |
| `position` | `String` | `before` | all |  |
| `full_name` | `String` | `before` | all |  |
| `first_name` | `String` | `before` | all |  |
| `last_name` | `String` | `before` | all |  |
| `report_primary_injury` | `String` | `before` | all |  |
| `report_secondary_injury` | `String` | `before` | all |  |
| `report_status` | `String` | `before` | all | Out / Doubtful / Questionable. Drives Layer 3 redistribution. |
| `practice_primary_injury` | `String` | `before` | all |  |
| `practice_secondary_injury` | `String` | `before` | all |  |
| `practice_status` | `String` | `before` | all |  |
| `date_modified` | `Datetime(time_unit='us', time_zone='UTC')` | `before` | 2016–2024 | Real UTC timestamp. The ONLY nflverse field that supports exact as-of filtering without a lag heuristic. |
| `season_type` | `String` | `before` | 2025 |  |

### `participation`

Per-play personnel: `offense_players` lists the 11 gsis_ids on the field. Joined to pbp dropbacks this yields pass-play participation, our routes-run proxy. See README limitations.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `nflverse_game_id` | `String` | `after` | all |  |
| `old_game_id` | `String` | `after` | all |  |
| `play_id` | `Float64` | `after` | all |  |
| `possession_team` | `String` | `after` | all |  |
| `offense_formation` | `String` | `after` | all |  |
| `offense_personnel` | `String` | `after` | all | e.g. '1 RB, 1 TE, 3 WR'. Personnel context for shares. |
| `defenders_in_box` | `Int32` | `after` | all |  |
| `defense_personnel` | `String` | `after` | all |  |
| `number_of_pass_rushers` | `Int32` | `after` | all |  |
| `players_on_play` | `String` | `after` | all |  |
| `offense_players` | `String` | `after` | all | Semicolon-delimited gsis_ids of the 11 offensive players on the play. Filtered to dropbacks this is the routes-run PROXY (Layer 3 exposure). Verified to join pbp pass plays 1:1 for 2024. |
| `defense_players` | `String` | `after` | all |  |
| `n_offense` | `Int32` | `after` | all |  |
| `n_defense` | `Int32` | `after` | all |  |
| `ngs_air_yards` | `Float64` | `after` | all |  |
| `time_to_throw` | `Float64` | `after` | all |  |
| `was_pressure` | `Boolean` | `after` | all |  |
| `route` | `String` | `after` | all | ONE route label per play, not per player — cannot be used to count routes run by individual receivers. |
| `defense_man_zone_type` | `String` | `after` | all |  |
| `defense_coverage_type` | `String` | `after` | all |  |
| `offense_names` | `String` | `after` | 2023–2025 |  |
| `defense_names` | `String` | `after` | 2023–2025 |  |
| `offense_positions` | `String` | `after` | 2023–2025 |  |
| `defense_positions` | `String` | `after` | 2023–2025 |  |
| `offense_numbers` | `String` | `after` | 2023–2025 |  |
| `defense_numbers` | `String` | `after` | 2023–2025 |  |

### `pbp`

Play-by-play. Source of team plays, pass/rush split, PROE (xpass/pass_oe), targets and carries by player.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `play_id` | `Float64` | `after` | all |  |
| `game_id` | `String` | `after` | all |  |
| `old_game_id` | `String` | `after` | all |  |
| `home_team` | `String` | `after` | all |  |
| `away_team` | `String` | `after` | all |  |
| `season_type` | `String` | `after` | all |  |
| `week` | `Int32` | `after` | all |  |
| `posteam` | `String` | `after` | all |  |
| `posteam_type` | `String` | `after` | all |  |
| `defteam` | `String` | `after` | all |  |
| `side_of_field` | `String` | `after` | all |  |
| `yardline_100` | `Float64` | `after` | all |  |
| `game_date` | `String` | `after` | all |  |
| `quarter_seconds_remaining` | `Float64` | `after` | all |  |
| `half_seconds_remaining` | `Float64` | `after` | all |  |
| `game_seconds_remaining` | `Float64` | `after` | all |  |
| `game_half` | `String` | `after` | all |  |
| `quarter_end` | `Float64` | `after` | all |  |
| `drive` | `Float64` | `after` | all |  |
| `sp` | `Float64` | `after` | all |  |
| `qtr` | `Float64` | `after` | all |  |
| `down` | `Float64` | `after` | all |  |
| `goal_to_go` | `Float64` | `after` | all |  |
| `time` | `String` | `after` | all |  |
| `yrdln` | `String` | `after` | all |  |
| `ydstogo` | `Float64` | `after` | all |  |
| `ydsnet` | `Float64` | `after` | all |  |
| `desc` | `String` | `after` | all |  |
| `play_type` | `String` | `after` | all |  |
| `yards_gained` | `Float64` | `after` | all |  |
| `shotgun` | `Float64` | `after` | all |  |
| `no_huddle` | `Float64` | `after` | all |  |
| `qb_dropback` | `Float64` | `after` | all |  |
| `qb_kneel` | `Float64` | `after` | all |  |
| `qb_spike` | `Float64` | `after` | all |  |
| `qb_scramble` | `Float64` | `after` | all |  |
| `pass_length` | `String` | `after` | all |  |
| `pass_location` | `String` | `after` | all |  |
| `air_yards` | `Float64` | `after` | all |  |
| `yards_after_catch` | `Float64` | `after` | all |  |
| `run_location` | `String` | `after` | all |  |
| `run_gap` | `String` | `after` | all |  |
| `field_goal_result` | `String` | `after` | all |  |
| `kick_distance` | `Float64` | `after` | all |  |
| `extra_point_result` | `String` | `after` | all |  |
| `two_point_conv_result` | `String` | `after` | all |  |
| `home_timeouts_remaining` | `Float64` | `after` | all |  |
| `away_timeouts_remaining` | `Float64` | `after` | all |  |
| `timeout` | `Float64` | `after` | all |  |
| `timeout_team` | `String` | `after` | all |  |
| `td_team` | `String` | `after` | all |  |
| `td_player_name` | `String` | `after` | all |  |
| `td_player_id` | `String` | `after` | all |  |
| `posteam_timeouts_remaining` | `Float64` | `after` | all |  |
| `defteam_timeouts_remaining` | `Float64` | `after` | all |  |
| `total_home_score` | `Float64` | `after` | all |  |
| `total_away_score` | `Float64` | `after` | all |  |
| `posteam_score` | `Float64` | `after` | all |  |
| `defteam_score` | `Float64` | `after` | all |  |
| `score_differential` | `Float64` | `after` | all |  |
| `posteam_score_post` | `Float64` | `after` | all |  |
| `defteam_score_post` | `Float64` | `after` | all |  |
| `score_differential_post` | `Float64` | `after` | all |  |
| `no_score_prob` | `Float64` | `after` | all |  |
| `opp_fg_prob` | `Float64` | `after` | all |  |
| `opp_safety_prob` | `Float64` | `after` | all |  |
| `opp_td_prob` | `Float64` | `after` | all |  |
| `fg_prob` | `Float64` | `after` | all |  |
| `safety_prob` | `Float64` | `after` | all |  |
| `td_prob` | `Float64` | `after` | all |  |
| `extra_point_prob` | `Float64` | `after` | all |  |
| `two_point_conversion_prob` | `Float64` | `after` | all |  |
| `ep` | `Float64` | `after` | all |  |
| `epa` | `Float64` | `after` | all |  |
| `total_home_epa` | `Float64` | `after` | all |  |
| `total_away_epa` | `Float64` | `after` | all |  |
| `total_home_rush_epa` | `Float64` | `after` | all |  |
| `total_away_rush_epa` | `Float64` | `after` | all |  |
| `total_home_pass_epa` | `Float64` | `after` | all |  |
| `total_away_pass_epa` | `Float64` | `after` | all |  |
| `air_epa` | `Float64` | `after` | all |  |
| `yac_epa` | `Float64` | `after` | all |  |
| `comp_air_epa` | `Float64` | `after` | all |  |
| `comp_yac_epa` | `Float64` | `after` | all |  |
| `total_home_comp_air_epa` | `Float64` | `after` | all |  |
| `total_away_comp_air_epa` | `Float64` | `after` | all |  |
| `total_home_comp_yac_epa` | `Float64` | `after` | all |  |
| `total_away_comp_yac_epa` | `Float64` | `after` | all |  |
| `total_home_raw_air_epa` | `Float64` | `after` | all |  |
| `total_away_raw_air_epa` | `Float64` | `after` | all |  |
| `total_home_raw_yac_epa` | `Float64` | `after` | all |  |
| `total_away_raw_yac_epa` | `Float64` | `after` | all |  |
| `wp` | `Float64` | `after` | all |  |
| `def_wp` | `Float64` | `after` | all |  |
| `home_wp` | `Float64` | `after` | all |  |
| `away_wp` | `Float64` | `after` | all |  |
| `wpa` | `Float64` | `after` | all |  |
| `vegas_wpa` | `Float64` | `after` | all |  |
| `vegas_home_wpa` | `Float64` | `after` | all |  |
| `home_wp_post` | `Float64` | `after` | all |  |
| `away_wp_post` | `Float64` | `after` | all |  |
| `vegas_wp` | `Float64` | `after` | all |  |
| `vegas_home_wp` | `Float64` | `after` | all |  |
| `total_home_rush_wpa` | `Float64` | `after` | all |  |
| `total_away_rush_wpa` | `Float64` | `after` | all |  |
| `total_home_pass_wpa` | `Float64` | `after` | all |  |
| `total_away_pass_wpa` | `Float64` | `after` | all |  |
| `air_wpa` | `Float64` | `after` | all |  |
| `yac_wpa` | `Float64` | `after` | all |  |
| `comp_air_wpa` | `Float64` | `after` | all |  |
| `comp_yac_wpa` | `Float64` | `after` | all |  |
| `total_home_comp_air_wpa` | `Float64` | `after` | all |  |
| `total_away_comp_air_wpa` | `Float64` | `after` | all |  |
| `total_home_comp_yac_wpa` | `Float64` | `after` | all |  |
| `total_away_comp_yac_wpa` | `Float64` | `after` | all |  |
| `total_home_raw_air_wpa` | `Float64` | `after` | all |  |
| `total_away_raw_air_wpa` | `Float64` | `after` | all |  |
| `total_home_raw_yac_wpa` | `Float64` | `after` | all |  |
| `total_away_raw_yac_wpa` | `Float64` | `after` | all |  |
| `punt_blocked` | `Float64` | `after` | all |  |
| `first_down_rush` | `Float64` | `after` | all |  |
| `first_down_pass` | `Float64` | `after` | all |  |
| `first_down_penalty` | `Float64` | `after` | all |  |
| `third_down_converted` | `Float64` | `after` | all |  |
| `third_down_failed` | `Float64` | `after` | all |  |
| `fourth_down_converted` | `Float64` | `after` | all |  |
| `fourth_down_failed` | `Float64` | `after` | all |  |
| `incomplete_pass` | `Float64` | `after` | all |  |
| `touchback` | `Float64` | `after` | all |  |
| `interception` | `Float64` | `after` | all |  |
| `punt_inside_twenty` | `Float64` | `after` | all |  |
| `punt_in_endzone` | `Float64` | `after` | all |  |
| `punt_out_of_bounds` | `Float64` | `after` | all |  |
| `punt_downed` | `Float64` | `after` | all |  |
| `punt_fair_catch` | `Float64` | `after` | all |  |
| `kickoff_inside_twenty` | `Float64` | `after` | all |  |
| `kickoff_in_endzone` | `Float64` | `after` | all |  |
| `kickoff_out_of_bounds` | `Float64` | `after` | all |  |
| `kickoff_downed` | `Float64` | `after` | all |  |
| `kickoff_fair_catch` | `Float64` | `after` | all |  |
| `fumble_forced` | `Float64` | `after` | all |  |
| `fumble_not_forced` | `Float64` | `after` | all |  |
| `fumble_out_of_bounds` | `Float64` | `after` | all |  |
| `solo_tackle` | `Float64` | `after` | all |  |
| `safety` | `Float64` | `after` | all |  |
| `penalty` | `Float64` | `after` | all |  |
| `tackled_for_loss` | `Float64` | `after` | all |  |
| `fumble_lost` | `Float64` | `after` | all |  |
| `own_kickoff_recovery` | `Float64` | `after` | all |  |
| `own_kickoff_recovery_td` | `Float64` | `after` | all |  |
| `qb_hit` | `Float64` | `after` | all |  |
| `rush_attempt` | `Float64` | `after` | all |  |
| `pass_attempt` | `Float64` | `after` | all |  |
| `sack` | `Float64` | `after` | all |  |
| `touchdown` | `Float64` | `after` | all |  |
| `pass_touchdown` | `Float64` | `after` | all |  |
| `rush_touchdown` | `Float64` | `after` | all |  |
| `return_touchdown` | `Float64` | `after` | all |  |
| `extra_point_attempt` | `Float64` | `after` | all |  |
| `two_point_attempt` | `Float64` | `after` | all |  |
| `field_goal_attempt` | `Float64` | `after` | all |  |
| `kickoff_attempt` | `Float64` | `after` | all |  |
| `punt_attempt` | `Float64` | `after` | all |  |
| `fumble` | `Float64` | `after` | all |  |
| `complete_pass` | `Float64` | `after` | all | 1 on a completed pass. Numerator of catch rate (Layer 4). |
| `assist_tackle` | `Float64` | `after` | all |  |
| `lateral_reception` | `Float64` | `after` | all |  |
| `lateral_rush` | `Float64` | `after` | all |  |
| `lateral_return` | `Float64` | `after` | all |  |
| `lateral_recovery` | `Float64` | `after` | all |  |
| `passer_player_id` | `String` | `after` | all |  |
| `passer_player_name` | `String` | `after` | all |  |
| `passing_yards` | `Float64` | `after` | all |  |
| `receiver_player_id` | `String` | `after` | all | Targeted receiver (gsis). Numerator of target share (Layer 3). |
| `receiver_player_name` | `String` | `after` | all |  |
| `receiving_yards` | `Float64` | `after` | all |  |
| `rusher_player_id` | `String` | `after` | all | Ball carrier (gsis). Numerator of carry share (Layer 3). |
| `rusher_player_name` | `String` | `after` | all |  |
| `rushing_yards` | `Float64` | `after` | all |  |
| `lateral_receiver_player_id` | `String` | `after` | all |  |
| `lateral_receiver_player_name` | `String` | `after` | all |  |
| `lateral_receiving_yards` | `Float64` | `after` | all |  |
| `lateral_rusher_player_id` | `String` | `after` | all |  |
| `lateral_rusher_player_name` | `String` | `after` | all |  |
| `lateral_rushing_yards` | `Float64` | `after` | all |  |
| `lateral_sack_player_id` | `String` | `after` | all |  |
| `lateral_sack_player_name` | `String` | `after` | all |  |
| `interception_player_id` | `String` | `after` | all |  |
| `interception_player_name` | `String` | `after` | all |  |
| `lateral_interception_player_id` | `String` | `after` | all |  |
| `lateral_interception_player_name` | `String` | `after` | all |  |
| `punt_returner_player_id` | `String` | `after` | all |  |
| `punt_returner_player_name` | `String` | `after` | all |  |
| `lateral_punt_returner_player_id` | `String` | `after` | all |  |
| `lateral_punt_returner_player_name` | `String` | `after` | all |  |
| `kickoff_returner_player_name` | `String` | `after` | all |  |
| `kickoff_returner_player_id` | `String` | `after` | all |  |
| `lateral_kickoff_returner_player_id` | `String` | `after` | all |  |
| `lateral_kickoff_returner_player_name` | `String` | `after` | all |  |
| `punter_player_id` | `String` | `after` | all |  |
| `punter_player_name` | `String` | `after` | all |  |
| `kicker_player_name` | `String` | `after` | all |  |
| `kicker_player_id` | `String` | `after` | all |  |
| `own_kickoff_recovery_player_id` | `String` | `after` | all |  |
| `own_kickoff_recovery_player_name` | `String` | `after` | all |  |
| `blocked_player_id` | `String` | `after` | all |  |
| `blocked_player_name` | `String` | `after` | all |  |
| `tackle_for_loss_1_player_id` | `String` | `after` | all |  |
| `tackle_for_loss_1_player_name` | `String` | `after` | all |  |
| `tackle_for_loss_2_player_id` | `String` | `after` | all |  |
| `tackle_for_loss_2_player_name` | `String` | `after` | all |  |
| `qb_hit_1_player_id` | `String` | `after` | all |  |
| `qb_hit_1_player_name` | `String` | `after` | all |  |
| `qb_hit_2_player_id` | `String` | `after` | all |  |
| `qb_hit_2_player_name` | `String` | `after` | all |  |
| `forced_fumble_player_1_team` | `String` | `after` | all |  |
| `forced_fumble_player_1_player_id` | `String` | `after` | all |  |
| `forced_fumble_player_1_player_name` | `String` | `after` | all |  |
| `forced_fumble_player_2_team` | `String` | `after` | all |  |
| `forced_fumble_player_2_player_id` | `String` | `after` | all |  |
| `forced_fumble_player_2_player_name` | `String` | `after` | all |  |
| `solo_tackle_1_team` | `String` | `after` | all |  |
| `solo_tackle_2_team` | `String` | `after` | all |  |
| `solo_tackle_1_player_id` | `String` | `after` | all |  |
| `solo_tackle_2_player_id` | `String` | `after` | all |  |
| `solo_tackle_1_player_name` | `String` | `after` | all |  |
| `solo_tackle_2_player_name` | `String` | `after` | all |  |
| `assist_tackle_1_player_id` | `String` | `after` | all |  |
| `assist_tackle_1_player_name` | `String` | `after` | all |  |
| `assist_tackle_1_team` | `String` | `after` | all |  |
| `assist_tackle_2_player_id` | `String` | `after` | all |  |
| `assist_tackle_2_player_name` | `String` | `after` | all |  |
| `assist_tackle_2_team` | `String` | `after` | all |  |
| `assist_tackle_3_player_id` | `String` | `after` | all |  |
| `assist_tackle_3_player_name` | `String` | `after` | all |  |
| `assist_tackle_3_team` | `String` | `after` | all |  |
| `assist_tackle_4_player_id` | `String` | `after` | all |  |
| `assist_tackle_4_player_name` | `String` | `after` | all |  |
| `assist_tackle_4_team` | `String` | `after` | all |  |
| `tackle_with_assist` | `Float64` | `after` | all |  |
| `tackle_with_assist_1_player_id` | `String` | `after` | all |  |
| `tackle_with_assist_1_player_name` | `String` | `after` | all |  |
| `tackle_with_assist_1_team` | `String` | `after` | all |  |
| `tackle_with_assist_2_player_id` | `String` | `after` | all |  |
| `tackle_with_assist_2_player_name` | `String` | `after` | all |  |
| `tackle_with_assist_2_team` | `String` | `after` | all |  |
| `pass_defense_1_player_id` | `String` | `after` | all |  |
| `pass_defense_1_player_name` | `String` | `after` | all |  |
| `pass_defense_2_player_id` | `String` | `after` | all |  |
| `pass_defense_2_player_name` | `String` | `after` | all |  |
| `fumbled_1_team` | `String` | `after` | all |  |
| `fumbled_1_player_id` | `String` | `after` | all |  |
| `fumbled_1_player_name` | `String` | `after` | all |  |
| `fumbled_2_player_id` | `String` | `after` | all |  |
| `fumbled_2_player_name` | `String` | `after` | all |  |
| `fumbled_2_team` | `String` | `after` | all |  |
| `fumble_recovery_1_team` | `String` | `after` | all |  |
| `fumble_recovery_1_yards` | `Float64` | `after` | all |  |
| `fumble_recovery_1_player_id` | `String` | `after` | all |  |
| `fumble_recovery_1_player_name` | `String` | `after` | all |  |
| `fumble_recovery_2_team` | `String` | `after` | all |  |
| `fumble_recovery_2_yards` | `Float64` | `after` | all |  |
| `fumble_recovery_2_player_id` | `String` | `after` | all |  |
| `fumble_recovery_2_player_name` | `String` | `after` | all |  |
| `sack_player_id` | `String` | `after` | all |  |
| `sack_player_name` | `String` | `after` | all |  |
| `half_sack_1_player_id` | `String` | `after` | all |  |
| `half_sack_1_player_name` | `String` | `after` | all |  |
| `half_sack_2_player_id` | `String` | `after` | all |  |
| `half_sack_2_player_name` | `String` | `after` | all |  |
| `return_team` | `String` | `after` | all |  |
| `return_yards` | `Float64` | `after` | all |  |
| `penalty_team` | `String` | `after` | all |  |
| `penalty_player_id` | `String` | `after` | all |  |
| `penalty_player_name` | `String` | `after` | all |  |
| `penalty_yards` | `Float64` | `after` | all |  |
| `replay_or_challenge` | `Float64` | `after` | all |  |
| `replay_or_challenge_result` | `String` | `after` | all |  |
| `penalty_type` | `String` | `after` | all |  |
| `defensive_two_point_attempt` | `Float64` | `after` | all |  |
| `defensive_two_point_conv` | `Float64` | `after` | all |  |
| `defensive_extra_point_attempt` | `Float64` | `after` | all |  |
| `defensive_extra_point_conv` | `Float64` | `after` | all |  |
| `safety_player_name` | `String` | `after` | all |  |
| `safety_player_id` | `String` | `after` | all |  |
| `season` | `Int32` | `after` | all |  |
| `cp` | `Float64` | `after` | all |  |
| `cpoe` | `Float64` | `after` | all |  |
| `series` | `Float64` | `after` | all |  |
| `series_success` | `Float64` | `after` | all |  |
| `series_result` | `String` | `after` | all |  |
| `order_sequence` | `Float64` | `after` | all |  |
| `start_time` | `String` | `after` | all |  |
| `time_of_day` | `String` | `after` | all |  |
| `stadium` | `String` | `after` | all |  |
| `weather` | `String` | `after` | all |  |
| `nfl_api_id` | `String` | `after` | all |  |
| `play_clock` | `String` | `after` | all |  |
| `play_deleted` | `Float64` | `after` | all |  |
| `play_type_nfl` | `String` | `after` | all |  |
| `special_teams_play` | `Float64` | `after` | all |  |
| `st_play_type` | `String` | `after` | all |  |
| `end_clock_time` | `String` | `after` | all |  |
| `end_yard_line` | `String` | `after` | all |  |
| `fixed_drive` | `Float64` | `after` | all |  |
| `fixed_drive_result` | `String` | `after` | all |  |
| `drive_real_start_time` | `String` | `after` | all |  |
| `drive_play_count` | `Float64` | `after` | all |  |
| `drive_time_of_possession` | `String` | `after` | all |  |
| `drive_first_downs` | `Float64` | `after` | all |  |
| `drive_inside20` | `Float64` | `after` | all |  |
| `drive_ended_with_score` | `Float64` | `after` | all |  |
| `drive_quarter_start` | `Float64` | `after` | all |  |
| `drive_quarter_end` | `Float64` | `after` | all |  |
| `drive_yards_penalized` | `Float64` | `after` | all |  |
| `drive_start_transition` | `String` | `after` | all |  |
| `drive_end_transition` | `String` | `after` | all |  |
| `drive_game_clock_start` | `String` | `after` | all |  |
| `drive_game_clock_end` | `String` | `after` | all |  |
| `drive_start_yard_line` | `String` | `after` | all |  |
| `drive_end_yard_line` | `String` | `after` | all |  |
| `drive_play_id_started` | `Float64` | `after` | all |  |
| `drive_play_id_ended` | `Float64` | `after` | all |  |
| `away_score` | `Int32` | `after` | all |  |
| `home_score` | `Int32` | `after` | all |  |
| `location` | `String` | `after` | all |  |
| `result` | `Int32` | `after` | all |  |
| `total` | `Int32` | `after` | all |  |
| `spread_line` | `Float64` | `after` | all |  |
| `total_line` | `Float64` | `after` | all |  |
| `div_game` | `Int32` | `after` | all |  |
| `roof` | `String` | `after` | all |  |
| `surface` | `String` | `after` | all |  |
| `temp` | `Int32` | `after` | all |  |
| `wind` | `Int32` | `after` | all |  |
| `home_coach` | `String` | `after` | all |  |
| `away_coach` | `String` | `after` | all |  |
| `stadium_id` | `String` | `after` | all |  |
| `game_stadium` | `String` | `after` | all |  |
| `aborted_play` | `Float64` | `after` | all |  |
| `success` | `Float64` | `after` | all |  |
| `passer` | `String` | `after` | all |  |
| `passer_jersey_number` | `Int32` | `after` | all |  |
| `rusher` | `String` | `after` | all |  |
| `rusher_jersey_number` | `Int32` | `after` | all |  |
| `receiver` | `String` | `after` | all |  |
| `receiver_jersey_number` | `Int32` | `after` | all |  |
| `pass` | `Float64` | `after` | all |  |
| `rush` | `Float64` | `after` | all |  |
| `first_down` | `Float64` | `after` | all |  |
| `special` | `Float64` | `after` | all |  |
| `play` | `Float64` | `after` | all |  |
| `passer_id` | `String` | `after` | all |  |
| `rusher_id` | `String` | `after` | all |  |
| `receiver_id` | `String` | `after` | all |  |
| `name` | `String` | `after` | all |  |
| `jersey_number` | `Int32` | `after` | all |  |
| `id` | `String` | `after` | all |  |
| `fantasy_player_name` | `String` | `after` | all |  |
| `fantasy_player_id` | `String` | `after` | all |  |
| `fantasy` | `String` | `after` | all |  |
| `fantasy_id` | `String` | `after` | all |  |
| `out_of_bounds` | `Float64` | `after` | all |  |
| `home_opening_kickoff` | `Float64` | `after` | all |  |
| `qb_epa` | `Float64` | `after` | all |  |
| `xyac_epa` | `Float64` | `after` | all |  |
| `xyac_mean_yardage` | `Float64` | `after` | all |  |
| `xyac_median_yardage` | `Int32` | `after` | all |  |
| `xyac_success` | `Float64` | `after` | all |  |
| `xyac_fd` | `Float64` | `after` | all |  |
| `xpass` | `Float64` | `after` | all | Model-expected pass probability. `pass_oe` = pass − xpass gives PROE. |
| `pass_oe` | `Float64` | `after` | all | Pass rate over expected for the play. Aggregate for team PROE (Layer 2). |

### `player_stats`

Weekly per-player box score: targets, receptions, carries, yards.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `player_id` | `String` | `after` | all |  |
| `player_name` | `String` | `after` | all |  |
| `player_display_name` | `String` | `after` | all |  |
| `position` | `String` | `after` | all |  |
| `position_group` | `String` | `after` | all |  |
| `headshot_url` | `String` | `after` | all |  |
| `season` | `Int32` | `after` | all |  |
| `week` | `Int32` | `after` | all |  |
| `season_type` | `String` | `after` | all |  |
| `game_id` | `String` | `after` | all |  |
| `team` | `String` | `after` | all |  |
| `opponent_team` | `String` | `after` | all |  |
| `completions` | `Int32` | `after` | all |  |
| `attempts` | `Int32` | `after` | all |  |
| `passing_yards` | `Int32` | `after` | all |  |
| `passing_tds` | `Int32` | `after` | all |  |
| `passing_interceptions` | `Int32` | `after` | all |  |
| `sacks_suffered` | `Int32` | `after` | all |  |
| `sack_yards_lost` | `Int32` | `after` | all |  |
| `sack_fumbles` | `Int32` | `after` | all |  |
| `sack_fumbles_lost` | `Int32` | `after` | all |  |
| `passing_air_yards` | `Int32` | `after` | all |  |
| `passing_yards_after_catch` | `Int32` | `after` | all |  |
| `passing_first_downs` | `Int32` | `after` | all |  |
| `passing_epa` | `Float64` | `after` | all |  |
| `passing_cpoe` | `Float64` | `after` | all |  |
| `passing_2pt_conversions` | `Int32` | `after` | all |  |
| `pacr` | `Float64` | `after` | all |  |
| `passing_10` | `Int32` | `after` | all |  |
| `passing_16` | `Int32` | `after` | all |  |
| `passing_20` | `Int32` | `after` | all |  |
| `passing_40` | `Int32` | `after` | all |  |
| `carries` | `Int32` | `after` | all |  |
| `rushing_yards` | `Int32` | `after` | all |  |
| `rushing_tds` | `Int32` | `after` | all |  |
| `rushing_fumbles` | `Int32` | `after` | all |  |
| `rushing_fumbles_lost` | `Int32` | `after` | all |  |
| `rushing_first_downs` | `Int32` | `after` | all |  |
| `rushing_epa` | `Float64` | `after` | all |  |
| `rushing_2pt_conversions` | `Int32` | `after` | all |  |
| `rushing_10` | `Int32` | `after` | all |  |
| `rushing_12` | `Int32` | `after` | all |  |
| `rushing_20` | `Int32` | `after` | all |  |
| `rushing_40` | `Int32` | `after` | all |  |
| `receptions` | `Int32` | `after` | all |  |
| `targets` | `Int32` | `after` | all |  |
| `receiving_yards` | `Int32` | `after` | all |  |
| `receiving_tds` | `Int32` | `after` | all |  |
| `receiving_fumbles` | `Int32` | `after` | all |  |
| `receiving_fumbles_lost` | `Int32` | `after` | all |  |
| `receiving_air_yards` | `Int32` | `after` | all |  |
| `receiving_yards_after_catch` | `Int32` | `after` | all |  |
| `receiving_first_downs` | `Int32` | `after` | all |  |
| `receiving_epa` | `Float64` | `after` | all |  |
| `receiving_2pt_conversions` | `Int32` | `after` | all |  |
| `receiving_10` | `Int32` | `after` | all |  |
| `receiving_16` | `Int32` | `after` | all |  |
| `receiving_20` | `Int32` | `after` | all |  |
| `receiving_40` | `Int32` | `after` | all |  |
| `racr` | `Float64` | `after` | all |  |
| `target_share` | `Float64` | `after` | all |  |
| `air_yards_share` | `Float64` | `after` | all |  |
| `wopr` | `Float64` | `after` | all |  |
| `special_teams_tds` | `Int32` | `after` | all |  |
| `def_tackles_solo` | `Int32` | `after` | all |  |
| `def_tackles_with_assist` | `Int32` | `after` | all |  |
| `def_tackle_assists` | `Int32` | `after` | all |  |
| `def_tackles_for_loss` | `Int32` | `after` | all |  |
| `def_tackles_for_loss_yards` | `Int32` | `after` | all |  |
| `def_fumbles_forced` | `Int32` | `after` | all |  |
| `def_sacks` | `Float64` | `after` | all |  |
| `def_sack_yards` | `Float64` | `after` | all |  |
| `def_qb_hits` | `Int32` | `after` | all |  |
| `def_interceptions` | `Int32` | `after` | all |  |
| `def_interception_yards` | `Int32` | `after` | all |  |
| `def_pass_defended` | `Int32` | `after` | all |  |
| `def_tds` | `Int32` | `after` | all |  |
| `def_fumbles` | `Int32` | `after` | all |  |
| `def_safeties` | `Int32` | `after` | all |  |
| `def_punt_blocks` | `Int32` | `after` | all |  |
| `def_pat_blocks` | `Int32` | `after` | all |  |
| `def_fg_blocks` | `Int32` | `after` | all |  |
| `def_2pt_atts` | `Int32` | `after` | all |  |
| `def_2pt_made` | `Int32` | `after` | all |  |
| `misc_yards` | `Int32` | `after` | all |  |
| `fumble_recovery_own` | `Int32` | `after` | all |  |
| `fumble_recovery_yards_own` | `Int32` | `after` | all |  |
| `fumble_recovery_opp` | `Int32` | `after` | all |  |
| `fumble_recovery_yards_opp` | `Int32` | `after` | all |  |
| `fumble_recovery_tds` | `Int32` | `after` | all |  |
| `penalties` | `Int32` | `after` | all |  |
| `penalty_yards` | `Int32` | `after` | all |  |
| `fumbles_forced_by_opp` | `Int32` | `after` | all |  |
| `fumbles_not_forced` | `Int32` | `after` | all |  |
| `fumbles_out_of_bounds` | `Int32` | `after` | all |  |
| `fumbles_total` | `Int32` | `after` | all |  |
| `fumbles_lost_total` | `Int32` | `after` | all |  |
| `punt_returns` | `Int32` | `after` | all |  |
| `punt_return_yards` | `Int32` | `after` | all |  |
| `kickoff_returns` | `Int32` | `after` | all |  |
| `kickoff_return_yards` | `Int32` | `after` | all |  |
| `fg_made` | `Int32` | `after` | all |  |
| `fg_att` | `Int32` | `after` | all |  |
| `fg_missed` | `Int32` | `after` | all |  |
| `fg_blocked` | `Int32` | `after` | all |  |
| `fg_long` | `Int32` | `after` | all |  |
| `fg_pct` | `Float64` | `after` | all |  |
| `fg_made_0_19` | `Int32` | `after` | all |  |
| `fg_made_20_29` | `Int32` | `after` | all |  |
| `fg_made_30_39` | `Int32` | `after` | all |  |
| `fg_made_40_49` | `Int32` | `after` | all |  |
| `fg_made_50_59` | `Int32` | `after` | all |  |
| `fg_made_60_` | `Int32` | `after` | all |  |
| `fg_missed_0_19` | `Int32` | `after` | all |  |
| `fg_missed_20_29` | `Int32` | `after` | all |  |
| `fg_missed_30_39` | `Int32` | `after` | all |  |
| `fg_missed_40_49` | `Int32` | `after` | all |  |
| `fg_missed_50_59` | `Int32` | `after` | all |  |
| `fg_missed_60_` | `Int32` | `after` | all |  |
| `fg_made_list` | `String` | `after` | all |  |
| `fg_missed_list` | `String` | `after` | all |  |
| `fg_blocked_list` | `String` | `after` | all |  |
| `fg_made_distance` | `Int32` | `after` | all |  |
| `fg_missed_distance` | `Int32` | `after` | all |  |
| `fg_blocked_distance` | `Int32` | `after` | all |  |
| `pat_made` | `Int32` | `after` | all |  |
| `pat_att` | `Int32` | `after` | all |  |
| `pat_missed` | `Int32` | `after` | all |  |
| `pat_blocked` | `Int32` | `after` | all |  |
| `pat_pct` | `Float64` | `after` | all |  |
| `gwfg_made` | `Int32` | `after` | all |  |
| `gwfg_att` | `Int32` | `after` | all |  |
| `gwfg_missed` | `Int32` | `after` | all |  |
| `gwfg_blocked` | `Int32` | `after` | all |  |
| `gwfg_distance` | `Int32` | `after` | all |  |
| `pt_att` | `Int32` | `after` | all |  |
| `pt_blocked` | `Int32` | `after` | all |  |
| `pt_long` | `Int32` | `after` | all |  |
| `pt_yards` | `Int32` | `after` | all |  |
| `pt_inside_20` | `Int32` | `after` | all |  |
| `pt_out_of_bounds` | `Int32` | `after` | all |  |
| `pt_downed` | `Int32` | `after` | all |  |
| `pt_touchback` | `Int32` | `after` | all |  |
| `pt_fair_caught` | `Int32` | `after` | all |  |
| `pt_returned` | `Int32` | `after` | all |  |
| `pt_return_yards` | `Int32` | `after` | all |  |
| `pt_return_tds` | `Int32` | `after` | all |  |
| `pt_net_yards` | `Int32` | `after` | all |  |
| `fantasy_points` | `Float64` | `after` | all |  |
| `fantasy_points_ppr` | `Float64` | `after` | all |  |

### `players`

Player ID crosswalk (gsis/pfr/pff/espn/ngs) plus biographical fields.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `gsis_id` | `String` | `static` | all |  |
| `display_name` | `String` | `before` | all |  |
| `common_first_name` | `String` | `before` | all |  |
| `first_name` | `String` | `before` | all |  |
| `last_name` | `String` | `before` | all |  |
| `short_name` | `String` | `before` | all |  |
| `football_name` | `String` | `before` | all |  |
| `suffix` | `String` | `before` | all |  |
| `esb_id` | `String` | `static` | all |  |
| `nfl_id` | `String` | `static` | all |  |
| `pfr_id` | `String` | `static` | all |  |
| `pff_id` | `String` | `static` | all |  |
| `otc_id` | `String` | `static` | all |  |
| `espn_id` | `String` | `static` | all |  |
| `smart_id` | `String` | `static` | all |  |
| `birth_date` | `String` | `static` | all |  |
| `position_group` | `String` | `before` | all |  |
| `position` | `String` | `before` | all |  |
| `ngs_position_group` | `String` | `before` | all |  |
| `ngs_position` | `String` | `before` | all |  |
| `height` | `Int32` | `static` | all |  |
| `weight` | `Int32` | `static` | all |  |
| `headshot` | `String` | `before` | all |  |
| `college_name` | `String` | `static` | all |  |
| `college_conference` | `String` | `static` | all |  |
| `jersey_number` | `String` | `before` | all |  |
| `rookie_season` | `Int32` | `static` | all |  |
| `last_season` | `Int32` | `before` | all |  |
| `latest_team` | `String` | `before` | all |  |
| `status` | `String` | `before` | all |  |
| `ngs_status` | `String` | `before` | all |  |
| `ngs_status_short_description` | `String` | `before` | all |  |
| `years_of_experience` | `Int32` | `before` | all |  |
| `pff_position` | `String` | `before` | all |  |
| `pff_status` | `String` | `before` | all |  |
| `draft_year` | `Int32` | `static` | all |  |
| `draft_round` | `Int32` | `static` | all |  |
| `draft_pick` | `Int32` | `static` | all |  |
| `draft_team` | `String` | `static` | all |  |

### `rosters_weekly`

Weekly roster status (active/inactive/IR) per player.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `season` | `Int32` | `lagged` | all |  |
| `team` | `String` | `lagged` | all |  |
| `position` | `String` | `lagged` | all |  |
| `depth_chart_position` | `String` | `lagged` | all |  |
| `jersey_number` | `Int32` | `lagged` | all |  |
| `status` | `String` | `lagged` | all |  |
| `full_name` | `String` | `lagged` | all |  |
| `first_name` | `String` | `lagged` | all |  |
| `last_name` | `String` | `lagged` | all |  |
| `birth_date` | `Date` | `lagged` | all |  |
| `height` | `Int32` | `lagged` | all |  |
| `weight` | `Int32` | `lagged` | all |  |
| `college` | `String` | `lagged` | all |  |
| `gsis_id` | `String` | `lagged` | all |  |
| `espn_id` | `String` | `lagged` | all |  |
| `sportradar_id` | `String` | `lagged` | all |  |
| `yahoo_id` | `String` | `lagged` | all |  |
| `rotowire_id` | `String` | `lagged` | all |  |
| `pff_id` | `String` | `lagged` | all |  |
| `pfr_id` | `String` | `lagged` | all |  |
| `fantasy_data_id` | `String` | `lagged` | all |  |
| `sleeper_id` | `String` | `lagged` | all |  |
| `years_exp` | `Int32` | `lagged` | all |  |
| `headshot_url` | `String` | `lagged` | all |  |
| `ngs_position` | `String` | `lagged` | all |  |
| `week` | `Int32` | `lagged` | all |  |
| `game_type` | `String` | `lagged` | all |  |
| `status_description_abbr` | `String` | `lagged` | all |  |
| `football_name` | `String` | `lagged` | all |  |
| `esb_id` | `String` | `lagged` | all |  |
| `gsis_it_id` | `String` | `lagged` | all |  |
| `smart_id` | `String` | `lagged` | all |  |
| `entry_year` | `Int32` | `lagged` | all |  |
| `rookie_year` | `Int32` | `lagged` | all |  |
| `draft_club` | `String` | `lagged` | all |  |
| `draft_number` | `Int32` | `lagged` | all |  |

### `schedules`

Game-level schedule with kickoff date/time, closing spread and total, roof/surface, rest days. Pre-game fields are known before kickoff; score/result columns are not.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `game_id` | `String` | `before` | all |  |
| `season` | `Int32` | `before` | all |  |
| `game_type` | `String` | `before` | all |  |
| `week` | `Int32` | `before` | all |  |
| `gameday` | `String` | `before` | all |  |
| `weekday` | `String` | `before` | all |  |
| `gametime` | `String` | `before` | all |  |
| `away_team` | `String` | `before` | all |  |
| `away_score` | `Int32` | `after` | all |  |
| `home_team` | `String` | `before` | all |  |
| `home_score` | `Int32` | `after` | all |  |
| `location` | `String` | `before` | all |  |
| `result` | `Int32` | `after` | all |  |
| `total` | `Int32` | `after` | all |  |
| `overtime` | `Int32` | `after` | all |  |
| `old_game_id` | `String` | `before` | all |  |
| `gsis` | `Int32` | `before` | all |  |
| `nfl_detail_id` | `String` | `before` | all |  |
| `pfr` | `String` | `before` | all |  |
| `pff` | `Int32` | `before` | all |  |
| `espn` | `String` | `before` | all |  |
| `ftn` | `Int32` | `before` | all |  |
| `away_rest` | `Int32` | `before` | all | Days of rest — known pre-kickoff. |
| `home_rest` | `Int32` | `before` | all | Days of rest — known pre-kickoff. |
| `away_moneyline` | `Int32` | `before` | all |  |
| `home_moneyline` | `Int32` | `before` | all |  |
| `spread_line` | `Float64` | `before` | all | Closing spread from the home team's perspective. Drives Layer 2 (pass/rush split). Note this is the CLOSING line — using it as a feature is fine for modelling but it is not what you would have had on Wednesday. |
| `away_spread_odds` | `Int32` | `before` | all |  |
| `home_spread_odds` | `Int32` | `before` | all |  |
| `total_line` | `Float64` | `before` | all | Closing game total. Drives Layer 1 (team plays). |
| `under_odds` | `Int32` | `before` | all |  |
| `over_odds` | `Int32` | `before` | all |  |
| `div_game` | `Int32` | `before` | all |  |
| `roof` | `String` | `before` | all | Stadium roof state; pre-kickoff for fixed roofs. |
| `surface` | `String` | `before` | all |  |
| `temp` | `Int32` | `after` | all |  |
| `wind` | `Int32` | `after` | all |  |
| `away_qb_id` | `String` | `after` | all |  |
| `home_qb_id` | `String` | `after` | all |  |
| `away_qb_name` | `String` | `after` | all |  |
| `home_qb_name` | `String` | `after` | all |  |
| `away_coach` | `String` | `before` | all |  |
| `home_coach` | `String` | `before` | all |  |
| `referee` | `String` | `before` | all |  |
| `stadium_id` | `String` | `before` | all |  |
| `stadium` | `String` | `before` | all |  |
| `kickoff_utc` | `Datetime(time_unit='us', time_zone='UTC')` | `before` | all | DERIVED by this repo: gameday + gametime parsed as US/Eastern, converted to UTC. The as-of cutoff for every Stage 2 feature. |

### `snap_counts`

PFR snap counts per player-game (offense/defense/ST, count and pct). Keyed on pfr_player_id -- needs the players crosswalk to join to gsis_id.

| column | dtype | timing | seasons | notes |
| --- | --- | --- | --- | --- |
| `game_id` | `String` | `after` | all |  |
| `pfr_game_id` | `String` | `after` | all |  |
| `season` | `Int32` | `after` | all |  |
| `game_type` | `String` | `after` | all |  |
| `week` | `Int32` | `after` | all |  |
| `player` | `String` | `after` | all |  |
| `pfr_player_id` | `String` | `after` | all | PFR id — join via players.pfr_id to reach gsis_id. |
| `position` | `String` | `after` | all |  |
| `team` | `String` | `after` | all |  |
| `opponent` | `String` | `after` | all |  |
| `offense_snaps` | `Float64` | `after` | all |  |
| `offense_pct` | `Float64` | `after` | all | Share of team offensive snaps. The Stage 1 exposure term, superseded by pass-play participation for receptions. |
| `defense_snaps` | `Float64` | `after` | all |  |
| `defense_pct` | `Float64` | `after` | all |  |
| `st_snaps` | `Float64` | `after` | all |  |
| `st_pct` | `Float64` | `after` | all |  |
