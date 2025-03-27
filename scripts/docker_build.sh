#!/bin/bash
set -e

# Extract the version number from the Cargo.toml file
# The sed command searches for a line starting with "version ="
# and captures the version string enclosed in double quotes
VERSION=$(sed -n '/^version\s*=/ s/.*"\(.*\)".*/\1/p' Cargo.toml)
echo "The version is:$VERSION"

# Check if a build mode parameter is provided, default is -beta indicating a debug build
DOCKER_TAG="${VERSION}-beta"
if [ "$1" == "release" ]; then
    DOCKER_TAG="${VERSION}"
fi

# Build the Docker image
docker build -t "hash-rstar:${DOCKER_TAG}" .

# Tag the current version as latest
if [ "$1" == "release" ]; then
    docker tag "hash-rstar:${DOCKER_TAG}" "hash-rstar:latest"
fi
