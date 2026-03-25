# Preprocessing Pipeline

Requires OpenAI-compatible LLM server ([vLLM](https://docs.vllm.ai)) for semantic analysis.

### Setup

```sh
# Install project deps
uv sync

# Install vLLM in separate env (strict PyTorch/CUDA requirements)
uv venv ~/.venv-vllm --python 3.13 && source ~/.venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto   # or --torch-backend=cu128/cu129/cu130/etc

# Start vLLM (context window set here)
vllm serve google/gemma-3-27b-it \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

### Scripts

Run in order. Each script caches its output to `data/` and skips if already cached.

```sh
uv run preproc/download.py                  # download CodeChat-V2.0
uv run preproc/filter.py                    # filter to English + Python
uv run preproc/syntax.py                    # ruff & radon static analysis
uv run preproc/semantic.py --parallel 24    # LLM-as-a-judge evaluation
```

### Distributed Run (multi-VM)

Shard across N VMs — each processes 1/N of the rows independently. E.g., 2:

```sh
# VM 1
uv run preproc/semantics.py --shard 2/2 --parallel 24 --host http://localhost:8000 --model google/gemma-3-27b-it

# VM 2
uv run preproc/semantics.py --shard 2/2 --parallel 24 --host http://localhost:8000 --model google/gemma-3-27b-it

# After all shards finish, merge on any machine with all shard files:
uv run preproc/semantics.py --merge
```

### semantics.py flags

| Flag         | Default                 | Description                          |
| ------------ | ----------------------- | ------------------------------------ |
| `--host`     | `http://localhost:8000` | vLLM server URL                      |
| `--model`    | `gemma3:27b`            | Model name (must match served model) |
| `--parallel` | `1`                     | Concurrent requests to LLM server    |
| `--shard`    | none                    | Shard spec `K/N` (e.g. `1/4`)        |
| `--merge`    | off                     | Merge shard outputs and exit         |
| `--sample`   | all                     | Random sample size                   |
| `--seed`     | `42`                    | Random seed for sampling             |
