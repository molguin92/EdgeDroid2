IMG_NAME = experiment_client
SOURCES = $(wildcard **/*.py)

.PHONY: image
image: Dockerfile $(SOURCES)
	docker build -t $(IMG_NAME) -f $< .
