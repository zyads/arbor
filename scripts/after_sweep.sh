#!/usr/bin/env bash
# Wait for the H2 sweep to finish, then immediately run the seed noise floor.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/sweep.sh" >/dev/null; do sleep 60; done
bash scripts/noise_floor.sh
