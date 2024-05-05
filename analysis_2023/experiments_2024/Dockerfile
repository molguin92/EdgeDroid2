FROM molguin/edgedroid2:experiment-client
LABEL authors="molguin"

RUN apt-get update && apt-get install -y iproute2
COPY inner/run_experiment.sh /opt/run_experiment.sh
ENTRYPOINT ["/usr/bin/env", "bash", "/opt/run_experiment.sh"]
