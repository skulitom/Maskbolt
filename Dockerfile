# Maskbolt's MCP server in a Linux container, for MCP catalogs that build servers from a Dockerfile
# (Glama, Docker's MCP Catalog) and for anyone who prefers Docker to a release archive.
#
#   docker build -t maskbolt .
#   docker run -i --rm -v "$PWD:/data" -v maskbolt-models:/cache maskbolt
#
# The server reads and writes only inside /data (mount your image folder there). The SAM model (375 MB) downloads
# on first use into /cache; the named volume keeps it between runs. `docker run --rm maskbolt version` prints the
# version, and any other CLI command works the same way.

# The toolchain the release builds use (.github/workflows/release.yml); Cargo.toml's rust-version is the minimum.
FROM rust:1.98.1-slim-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
RUN cargo build --release --locked --bin maskbolt

FROM debian:trixie-slim
# HTTPS for the model download uses rustls with built-in root certificates, so the image needs no extra packages.
RUN useradd --create-home --uid 10001 maskbolt \
    && mkdir -p /data /cache \
    && chown maskbolt:maskbolt /data /cache
COPY --from=build /src/target/release/maskbolt /usr/local/bin/maskbolt
COPY LICENSE THIRD_PARTY.txt /usr/share/doc/maskbolt/
USER maskbolt
ENV MASKBOLT_MODEL_DIR=/cache
WORKDIR /data
ENTRYPOINT ["maskbolt"]
CMD ["mcp", "--root", "/data"]
