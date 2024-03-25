FROM ubuntu:22.04 as builder
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
RUN apt update && apt install -y git build-essential postgresql-14 postgresql-server-dev-14
RUN git clone --branch v0.6.1 https://github.com/pgvector/pgvector.git
RUN cd pgvector && make && make install

FROM postgres:14-alpine
COPY --from=builder /usr/share/postgresql/14/extension/vector* /usr/local/share/postgresql/extension/
COPY --from=builder /usr/lib/postgresql/14/lib/vector.so /usr/local/lib/postgresql/vector.so
