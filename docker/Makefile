MKFILE_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
MNT_DIR := $(abspath $(MKFILE_DIR)../)

USER := $(shell id -u)
DOCKER_CMD         = docker run

# Main docker image name
DOCKER_IMG         = imagerecognition-build:0.1

# Where to mount the base of the repository in the container
DOCKER_MOUNT_POINT = /imageRecognition/

# Set up source code volume mount
DOCKER_VOLUME_MOUNT = -v $(MNT_DIR):$(DOCKER_MOUNT_POINT)

# Working directories
WORKING_DIRECTORY = $(DOCKER_MOUNT_POINT)
DOCKER_WD = /imageRecognition

# Create optional environment variable and user flags to be passed to the docker command
ENV_VARS =
USER_FLAGS =
LOCAL_USER = --user=$(USER):$(USER)

# Set up final docker commands
DOCKER_BASE = $(DOCKER_CMD) --rm -it $(USER_FLAGS) $(ENV_VARS) $(DOCKER_VOLUME_MOUNT) -w $(WORKING_DIRECTORY)
DOCKER = $(DOCKER_BASE) $(DOCKER_IMG)

echo:
	@echo $(MNT_DIR)

image:
	docker build . -t $(DOCKER_IMG)

annote_ocr:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/ocr/annote.sh

dataset_ocr:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/ocr/dataset.sh

train_ocr:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/ocr/train.sh

test:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /imageRecognition/docker/runtests.sh

shell:
	$(DOCKER) bash
