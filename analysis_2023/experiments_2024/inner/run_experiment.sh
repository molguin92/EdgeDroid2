#!/usr/bin/env bash
set -x

OUT_DIR="/opt/output"

# more latencies
# autocorrelation
tc qdisc add dev lo root netem delay "${LATENCY}ms" "${VARIANCE}ms" "${CORREL}%" distribution "${DIST:-"pareto"}"

python3 /opt/edgedroid-experiments/experiment_server.py --truncate "${NUM_STEPS}" -v -o "${OUT_DIR}" localhost 5000 "${TRACE}" &
python3 /opt/edgedroid-experiments/experiment_client.py --truncate "${NUM_STEPS}" -v -o "${OUT_DIR}" "${MODEL}" localhost 5000 "${TRACE}"
