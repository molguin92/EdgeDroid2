#!/usr/bin/env bash

if [ -z "$EXPERIMENT_ID" ]; then
  echo "EXPERIMENT_ID must be set."
  exit 1
fi

TRACE="${TRACE:-square00}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-"./output/$EXPERIMENT_ID"}"
NREPS="${NREPS:-20}"
PORT="${PORT:-5000}"

mkdir -p "$BASE_OUTPUT_DIR"

for i in $(eval echo "{1..$NREPS}"); do
  OUTPUT_DIR="$(printf "%s/rep%02d" "$BASE_OUTPUT_DIR" "$i")"
  mkdir -p "$OUTPUT_DIR"
  # language=docker
  COMPOSE="$(
    EXPERIMENT_ID="$EXPERIMENT_ID" \
    TRACE="$TRACE" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    PORT="$PORT" \
    envsubst << EOF # language=docker
version: "3"
networks:
  experiment: {}
services:
  server:
    image: molguin/edgedroid2:experiment-server
    entrypoint: >-
      bash -c
      "/bin/python3
      ./experiment_server.py
      0.0.0.0 $PORT $TRACE
      -o=/opt/output -v
      && sleep infinity"
    networks:
      experiment: {}
    volumes:
      - $OUTPUT_DIR:/opt/output/:rw
  client:
    image: molguin/edgedroid2:experiment-client
    command: ["$EXPERIMENT_ID", "server", "$PORT", "$TRACE", "-v", "-o=/opt/output"]
    networks:
      experiment: {}
    volumes:
      - $OUTPUT_DIR:/opt/output/:rw
    depends_on:
      - server
EOF
  )"

  docker compose -v -f - up --abort-on-container-exit <<< "$COMPOSE"
done
