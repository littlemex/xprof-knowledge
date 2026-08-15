# Reference container image for xprof-knowledge — a sample for running the MCP in a container
# (e.g. on Kubernetes). The package is the source of truth; this image just pip-installs it and
# runs the console script. No infrastructure or orchestrator is assumed.
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
RUN useradd -u 1000 -m app && chown -R app /app
USER app
ENV MCP_PORT=8080
EXPOSE 8080
CMD ["xprof-knowledge-mcp"]
