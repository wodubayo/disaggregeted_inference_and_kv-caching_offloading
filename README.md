
# legion-llm-mini — Disaggregated Inference + KV Offload (Single-GPU)

This is a **self-contained mini project** that emulates **disaggregated prefill/decode** and **KV-cache offloading** on a single Lenovo Legion GPU, **without touching your other projects**.

It uses **Docker Compose** with **vLLM** (for PagedAttention + LMCache) and a tiny **FastAPI router** that routes requests to separate **prefill** and **decode** services.
Both services share the same LMCache directory so prefill warms the cache and decode reuses it.

> This is a learning scaffold. Real disaggregation usually runs on separate GPU pools or nodes. On a single GPU laptop, the two services will share the device and time-slice, but the architecture and configs map 1:1 to bigger clusters later.

---

## What you get
- **Safe isolation** via Docker (no conda/env pollution, no global CUDA changes).
- **KV offload** to host RAM and disk using vLLM's LMCache + swap-space flags.
- **Router** with simple prefix-hash awareness to simulate cache-aware routing.
- Minimal, explicit **env files** and **volume mounts** under this project folder only.

## Prereqs
- NVIDIA driver + CUDA runtime (already on your Legion).
- Docker Desktop or Docker Engine with **nvidia-container-toolkit**.
- `docker compose version` ≥ 2.20.

## Quickstart
```bash
# From this project root:
./scripts/setup_dirs.sh      # creates local folders and permissions
docker compose pull          # pull images
docker compose up -d         # start router, prefill, decode

# Test
curl -s http://localhost:8088/healthz
curl -s http://localhost:8088/generate -H "Content-Type: application/json" -d '{
  "prompt": "Explain Rossby number in oceanography in 5 sentences.",
  "max_tokens": 128,
  "temperature": 0.2
}'
```

## Model selection
By default we set `MODEL_ID=meta-llama/Llama-3.1-8B-Instruct` (HF). You can change this in `configs/model.env`.
> Note: using an 8B model is recommended on a laptop GPU. Larger models may OOM.

## Folder layout
```
legion-llm-mini/
├─ docker-compose.yml
├─ .env
├─ README.md
├─ configs/
│  ├─ model.env
│  └─ router.env
├─ router/
│  ├─ app.py
│  └─ requirements.txt
├─ scripts/
│  ├─ setup_dirs.sh
│  └─ down.sh
├─ data/
│  └─ kv_cache/        # vLLM LMCache path (KV offload to disk)
└─ models/             # optional local models
```

## How disaggregation is emulated here
- **prefill** service: receives the initial prompt to build KV. (Our router sends the first call here.)
- **decode** service: receives follow-up generation using the same prefix; since both share `data/kv_cache`, vLLM can reuse KV pages from disk/host memory if beneficial.

**vLLM flags we use:**
- `--swap-space <GB>` → use host RAM for KV paging
- `--enable-lmcache` + `--lmcache-path /data/kv_cache` → on-disk KV paging

## Ports
- Router: `8088`
- Prefill vLLM OpenAI API: `8010`
- Decode vLLM OpenAI API: `8020`

## Clean-up
```bash
docker compose down
./scripts/down.sh       # removes volumes if you want a fresh slate
```

## Notes
- This doesn’t modify global CUDA or any of your other projects; everything stays inside this folder and Docker volumes.
- On first run, model weights will be downloaded into the `prefill` and `decode` containers' cache (~HF cache inside the container) unless you mount `models/` with your own weights.
