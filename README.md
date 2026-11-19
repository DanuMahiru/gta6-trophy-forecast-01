# gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DanuMahiru/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DanuMahiru/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of GTA VI's 55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate one week after launch (target 2026-11-26 00:00 UTC). Per trophy, in logit space: a straight-line trend fit on the last 24 hours of reads, slopes shrunk toward a pooled slope-vs-level line across all trophies, and a dilution shift when the tracker page goes stale. Standard library only, deterministic.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `/input` (or `$MAKEFUN_INPUT_DIR`) and writes one `*.out` per case to `/output` (or `$MAKEFUN_OUTPUT_DIR`).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
