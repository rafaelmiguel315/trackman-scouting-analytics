# Benchmark provenance and comparison rules

## Official MLB source
Baseball Savant Statcast Search and CSV documentation:
https://baseballsavant.mlb.com/statcast_search
https://baseballsavant.mlb.com/csv-docs
Pitch Movement leaderboard:
https://baseballsavant.mlb.com/leaderboard/pitch-movement

The MLB pitch movement leaderboard compares like pitch types, within +/-2 mph and +/-0.5 ft of extension/release for its movement-versus-comparable display. Induced movement excludes gravity; total movement includes it. Do not treat those displays as interchangeable.

## Important limitations
- MLB Statcast pitch tracking is not a substitute for TrackMan International Scouting or minor-league distributions.
- Do not publish MiLB/DSL, arm-slot-adjusted or 2025-2026 percentile grades without a licensed, traceable source and explicit metric/coordinate conversion.
- Statcast `pfx_x` is in feet from the catcher's perspective; `pfx_z` is also in feet. TrackMan HorzBreak/IVB commonly use inches; verify handedness, coordinate conventions and measurement definitions before comparison.
- `arm_angle` is a measured field in Statcast; never infer it from release height alone.
- MLB 2025 and MLB 2026 must be separated, as must level, pitch type, pitcher handedness, and arm-angle bucket if available.
- Small-sample observations must not be presented as reliable effectiveness grades.
- Historical player CSVs remain in Supabase; benchmark data must never overwrite them.

## Required benchmark record fields
`season,level,pitch_type,pitcher_hand,arm_angle_min,arm_angle_max,metric,unit,mean,sd,n_pitches,n_pitchers,source_url,source_retrieved_at`.

Missing or unmatched records should result in **No validated benchmark**. Red = good, yellow = average, blue = poor only when the metric direction is defensible for that matched pitch type and context. Pure IVB/HB magnitude has no universal 'higher is better' direction.

## Validation gate before deployment
1. CI must pass unit tests and generated-PDF integration tests.
2. Validate at least one actual de-identified TrackMan pitcher CSV and one batter CSV, including column variations.
3. Verify PDF output visually and check benchmark provenance.
4. Only then merge the PR to main and verify Streamlit deployment and data persistence.
