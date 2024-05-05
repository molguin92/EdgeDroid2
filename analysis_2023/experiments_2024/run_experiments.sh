#!/usr/bin/env bash

set -e

# SETTINGS
LATENCY=50  # one-way latency!
VARIANCE=10  # also one-way
CORREL=25
NUM_STEPS=100


REPS_PER_EXP=10
OUT_DIR="./output"
mkdir -p "${OUT_DIR}"

for MODEL in legacy first-order first-order-median curve-high-neuro curve-low-neuro; do
  EXP_OUT_DIR="${OUT_DIR}/${MODEL}"
  mkdir -p "${EXP_OUT_DIR}"
  docker run --rm -it --cap-add NET_ADMIN -v "${EXP_OUT_DIR}:/opt/output"\
    -e LATENCY=${LATENCY} -e VARIANCE=${VARIANCE} -e CORREL=${CORREL} -e \
    NUM_STEPS=${NUM_STEPS} -e TRACE=square00 -e MODEL=${MODEL} "experiment_client"
done