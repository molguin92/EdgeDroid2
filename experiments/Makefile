BUILD_CMD = docker buildx build --push --platform linux/arm64
DOCKER_USER = molguin
IMG_REPO = $(DOCKER_USER)/edgedroid2

SOURCES = $(wildcard **/*.py)

all: edgedroid-experiment-server edgedroid-experiment-client
.PHONY: all

edgedroid-experiment-client: Dockerfile $(SOURCES)
	$(BUILD_CMD) --target client -t $(IMG_REPO):experiment-client -f $< .

edgedroid-experiment-server: Dockerfile $(SOURCES)
	$(BUILD_CMD) --target server -t $(IMG_REPO):experiment-server -f $< .

clean:
	docker image rm $(IMG_REPO):experiment-client
	docker image rm $(IMG_REPO):experiment-server
