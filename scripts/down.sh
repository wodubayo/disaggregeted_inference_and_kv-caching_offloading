
#!/usr/bin/env bash
set -euo pipefail

docker compose down -v || true
echo "🧹 Removed containers and named volumes. 'data/kv_cache' folder remains on disk."
