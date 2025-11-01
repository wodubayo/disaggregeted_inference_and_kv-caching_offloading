
#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/kv_cache
mkdir -p models
chmod -R 777 data/kv_cache || true

echo "✅ Created data/kv_cache and models. Adjust permissions as needed."
