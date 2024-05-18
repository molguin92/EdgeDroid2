#!/usr/bin/env bash

set -ex
make image

# SETTINGS
LATENCY=100  # one-way latency! milliseconds
VARIANCE=20  # also one-way
NUM_STEPS=50
REPS_PER_EXP=10

for CORREL in 0 25 50 75 100; do
  OUT_DIR="./output/lat${LATENCY}_var${VARIANCE}_corr${CORREL}"
  mkdir -p "${OUT_DIR}"

  for REP in $(seq 1 ${REPS_PER_EXP}); do
    for MODEL in legacy first-order first-order-median curve-high-neuro curve-low-neuro; do
      EXP_OUT_DIR="${OUT_DIR}/${MODEL}_${REP}_of_${REPS_PER_EXP}"
      mkdir -p "${EXP_OUT_DIR}"
      docker run --rm -it --cap-add NET_ADMIN -v "${EXP_OUT_DIR}:/opt/output"\
        -e LATENCY=${LATENCY} -e VARIANCE=${VARIANCE} -e CORREL=${CORREL} \
        -e NUM_STEPS=${NUM_STEPS} -e TRACE=square00 -e MODEL=${MODEL} "experiment_client"
    done
  done
done

set +ex