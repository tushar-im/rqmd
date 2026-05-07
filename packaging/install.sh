#!/usr/bin/env bash
set -euo pipefail

DEST_DIR=${1:-$HOME/.local/bin}
BIN_PATH=${2:-target/release/rqmd}
mkdir -p "$DEST_DIR"
install -m 0755 "$BIN_PATH" "$DEST_DIR/rqmd"
echo "installed rqmd to $DEST_DIR/rqmd"
