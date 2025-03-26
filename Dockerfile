FROM rust:1.84.1 AS chef 
# We only pay the installation cost once, 
# it will be cached from the second build onwards
RUN cargo install cargo-chef 
WORKDIR app

FROM chef AS planner
COPY . .
RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
RUN apt-get update && apt-get install -y protobuf-compiler
# Build application
COPY . .

# 国内使用清华镜像源加速
RUN mkdir -p /usr/local/cargo/ && \
    echo '[source.crates-io]' > /usr/local/cargo/config.toml && \
    echo 'replace-with = "ustc"' >> /usr/local/cargo/config.toml && \
    echo '[source.ustc]' >> /usr/local/cargo/config.toml && \
    echo 'registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"' >> /usr/local/cargo/config.toml

RUN cargo build --release --bin hash_rstar

# We do not need the Rust toolchain to run the binary!
# 注意：不要使用 alpine。 alpine 缺少动态库 libm.so libc.so 导致服务运行失败
# 使用包含动态库的 debian 镜像
FROM debian:bookworm-slim AS runtime
WORKDIR /app
# 从构建阶段复制二进制文件
COPY --from=builder /app/target/release/hash_rstar /app/hash_rstar
# 设置启动命令
ENTRYPOINT ["/app/hash_rstar"]
