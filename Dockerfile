FROM python:3.12-bookworm

RUN apt update && apt install -y git make curl
RUN useradd -m -u 1000 user
# Put the venv on PATH so the entrypoint can call `python` directly
# without going through `uv run`. `uv run` re-validates the universal
# lockfile on every invocation which is multi-minute on this codebase.
ENV PATH="/home/user/.local/bin:/mteb/.venv/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Cache root for mteb. The leaderboard reads its parquet from
# `${MTEB_CACHE}/leaderboard/__cached_results.parquet`; the file is
# baked into the image further down. The directory is created (and
# chowned) up-front while still root so the unprivileged `user` step
# below can populate it. /cache is used (rather than the default
# `~/.cache/mteb`) because HF Spaces deployments inherit this image
# and a uid-independent path simplifies operations.
ENV MTEB_CACHE=/cache
RUN mkdir -p /cache/leaderboard && chown -R user:user /cache

# Copy the current directory contents into the container. .dockerignore
# keeps `.venv`, `.git`, tests, docs, etc. out of the build context.
COPY --chown=user:user . /mteb

USER user
WORKDIR /mteb

# Install dependencies with leaderboard extras. --frozen skips the
# universal lockfile re-validation; uv.lock is committed and authoritative.
RUN uv sync --frozen --extra leaderboard

# Bake the leaderboard parquet cache into the image. The leaderboard
# reads this file via DuckDB at boot, so it must be present before the
# container starts. Baking it in (rather than downloading at first
# boot) means container start is fast and offline-capable, and HF
# Spaces factory restarts don't re-pay the cost.
#
# The parquet is consumed from the build context root. It must be
# placed there ahead of `docker build` (it is gitignored, so CI
# pipelines need a step to populate it, e.g. by downloading from a
# canonical published location once that exists, or by running a
# `_load_from_cache(rebuild=True)` regen step).
COPY --chown=user:user __cached_results.parquet /cache/leaderboard/__cached_results.parquet

ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

CMD ["python", "-m", "mteb", "leaderboard"]
