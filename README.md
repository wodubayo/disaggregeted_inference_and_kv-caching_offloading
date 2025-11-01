
# Disaggregated Inference + KV Offload (Single-GPU)

This repository demonstrates a modular setup for **disaggregated inference**, where prefill and decode stages of an LLM pipeline are separated into individual backends. The architecture supports **KV-cache offloading, CPU/GPU memory partitioning**, and **router-level orchestration**, designed for experimentation on constrained hardware (e.g., RTX 2080, 16-32 GB RAM).

## 💡 Motivation & Hardware Context

This project was built to explore how large-language-model inference can be disaggregated across limited-resource devices — an approach usually reserved for enterprise-grade clusters.
It’s part of a broader effort to understand distributed AI system design and inference optimization at both the software and hardware levels.

I’m currently experimenting on a Lenovo Legion laptop configured with:

|Component	    |   Specification                         |
|---------------|-----------------------------------------|
|GPU	          | NVIDIA RTX 2080 Super (Max-Q, 8 GB VRAM)|
|CPU	          | Intel i7 – 10th Gen, 8 Cores            |
|Memory	        | 32 GB DDR4 RAM                          |
|Storage	      | 1 TB NVMe SSD                           |
|OS	            | Ubuntu 22.04 LTS                        |
|Docker Runtime | NVIDIA Container Toolkit + CUDA 12.x    |

Because the 2080 Super lacks the compute capability (≥ 8.0) required for advanced FlashAttention v2 and tensor-parallel inference, I implemented a lightweight, V0 inference engine configuration with router-mediated offloading.
The goal was to simulate large-scale inference splitting (prefill vs. decode) using a single-GPU environment with CPU KV-cache offloading — proving that efficient orchestration is possible even under tight memory constraints.

## 🚀 Features
- 🧩 Modular design — independent router, prefill, and (optional) decode services
- ⚙️ Docker Compose orchestration — single command brings up the whole stack
- 🧠 FastAPI router — acts as a lightweight load balancer and OpenAI-compatible proxy
- 💾 KV-cache offload — explore CPU or disk-based caching via mounted volumes
- 🔄 OpenAI API compatibility — supports /v1/completions, /v1/chat/completions, and /v1/models
- 🔍 Extendable — add Prometheus metrics, decoding backends, or caching policies later

## Prereqs
- NVIDIA driver + CUDA runtime (already on your Legion).
- Docker Desktop or Docker Engine with **nvidia-container-toolkit**.
- `docker compose version` ≥ 2.20.

## 🏗️ Architecture Overview
```bash
        ┌──────────────────────────────┐
        │          Router              │
        │  FastAPI + HTTPX Proxy       │
        │  - Routes /generate, /v1/*   │
        │  - Health checks             │
        └───────▲───────────┬──────────┘
                │           │
       ┌────────┘           └────────┐
┌────────────┐               ┌────────────┐
│  Prefill   │               │  Decode    │
│ vLLM OpenAI│               │ vLLM       │
│  /v1 API   │               │  /v1 API   │
└────────────┘               └────────────┘
   (GPU A)                       (GPU B)
```
In this repo, only the prefill backend is active by default.
## Quickstart
```bash
# 1️⃣ Clone & enter project
git clone https://github.com/wodubayo/disaggregated-inference-kv-caching.git
cd disaggregated-inference-kv-caching

# 2️⃣ Copy and edit environment variables
cp .env.example .env
# Optional: add HUGGING_FACE_HUB_TOKEN if needed

# 3️⃣ Build and start services
docker compose build router
docker compose up -d

# 4️⃣ Verify everything is live
curl -sS http://127.0.0.1:8088/healthz | jq .
curl -sS http://127.0.0.1:8088/v1/models | jq .
```

## 🧪 Example Queries
Text Completion
```bash
curl -sS http://127.0.0.1:8088/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "Explain the Rossby number in one line.",
    "max_tokens": 32
  }' | jq .
```
Chat completion
```bash
curl -sS http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hi."}
    ],
    "max_tokens": 16
  }' | jq .
```

## Repo Layout
```bash
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

## 🧠 Environment Variables
| Variable                 | Description                 | Default                      |
| ------------------------ | --------------------------- | ---------------------------- |
| `MODEL_ID`               | Hugging Face model ID       | `Qwen/Qwen2.5-1.5B-Instruct` |
| `MAX_MODEL_LEN`          | Max sequence length         | `2048`                       |
| `GPU_MEMORY_UTILIZATION` | Fraction of GPU VRAM to use | `0.55`                       |
| `SWAP_SPACE_GB`          | Swap allocation             | `8`                          |
| `CPU_OFFLOAD_GB`         | CPU memory for offload      | `8`                          |
| `PREFILL_PORT`           | vLLM backend port           | `8010`                       |
| `ROUTER_PORT`            | Router external port        | `8088`                       |

## Model selection
> Using an 8B model is the upper bound for consumer GPUs (~8 GB VRAM).  
> For smoother runs on limited memory, try `Qwen/Qwen2.5-1.5B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.3`.

You can change this in `configs/model.env`.
---

## How Disaggregation Is Emulated Here
- **Prefill service:** handles the initial prompt to build the key-value (KV) cache.  
- **Decode service:** performs follow-up generation using the same prefix.  
- Both containers share `/data/kv_cache`, allowing vLLM to reuse cached tensors and emulate inter-node KV reuse.

This setup does not perform true multi-node streaming yet; instead, it **simulates disaggregation via shared disk-based KV cache**.

---

## vLLM Flags Used
- `--swap-space <GB>` → enables host RAM paging for KV  
- `--enable-lmcache` + `--lmcache-path /data/kv_cache` → enables on-disk KV reuse  
- *(Optional)* `--gpu-memory-utilization 0.6` → limits GPU VRAM usage  
- *(Optional)* `--tensor-parallel-size 1` → single-GPU inference mode  

---

## Ports
| Service | Port | Role |
|----------|------|------|
| Router | 8088 | API Gateway |
| Prefill | 8010 | KV Construction |
| Decode | 8020 | Token Generation |

---

## Clean-Up
```bash
docker compose down           # stops containers
./scripts/down.sh             # optional: removes volumes & cache
```

## Notes
- Fully containerized — no global CUDA configuration changes.
- On first run, model weights are downloaded into the container’s cache (~HF cache).
- To persist weights:
  ```bash
  -v ./models:/root/.cache/huggingface
  ```