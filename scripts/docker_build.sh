#!/bin/bash
set -e

VERSION=$(sed -n '/^version\s*=/ s/.*"\(.*\)".*/\1/p' Cargo.toml)
echo "The version is:$VERSION"

# 检查是否提供了构建模式参数，默认为 -beta 表示 debug 构建
DOCKER_TAG="${VERSION}-beta"
if [ "$1" == "release" ]; then
    DOCKER_TAG="${VERSION}"
fi

# 构建 Docker 镜像
docker build -t "hash-rstar:${DOCKER_TAG}" .