DOCKER_USER = molguin
IMG_REPO = $(DOCKER_USER)/edgedroid2
PLATFORMS ?= linux/arm64,linux/amd64
BUILD_CMD = docker buildx build --push --platform $(PLATFORMS)

SOURCES := $(wildcard edgedroid/**/*.py)


all: edgedroid-base experiments
.PHONY: all clean dlib edgedroid-base experiments

#login:
#	docker login -u $(DOCKER_USER)

#server: Dockerfile
#	$(BUILD_CMD) --target server -t $(IMG_REPO):server -f $< .
#
#client: Dockerfile
#	$(BUILD_CMD) --target client -t $(IMG_REPO):client -f $< .

edgedroid-base: Dockerfile $(SOURCES)
	$(BUILD_CMD) -t $(IMG_REPO):latest -f $< .

experiments: edgedroid-base
	make -C experiments all

dlib: Dockerfile.dlib
	$(BUILD_CMD) -t $(IMG_REPO):dlib-base -f $< .

clean:
	docker image rm $(IMG_REPO):latest
