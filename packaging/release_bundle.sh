#!/usr/bin/env bash
set -euo pipefail

BIN_PATH=${1:-target/release/rqmd}
OUT_DIR=${2:-dist}
mkdir -p "$OUT_DIR"
cp "$BIN_PATH" "$OUT_DIR/rqmd"

(
  cd "$OUT_DIR"
  shasum -a 256 rqmd > sha256sums.txt
)

cat > "$OUT_DIR/sbom.json" <<JSON
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "version": 1,
  "components": [
    {
      "type": "application",
      "name": "rqmd",
      "version": "0.1.0",
      "hashes": [
        {"alg": "SHA-256", "content": "$(cut -d' ' -f1 "$OUT_DIR/sha256sums.txt")"}
      ]
    }
  ]
}
JSON

echo "bundle written to $OUT_DIR"
