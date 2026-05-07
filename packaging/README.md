# Packaging checklist

## Build + artifact generation

- Build targets: x86_64/aarch64 for macOS/Linux/Windows
- Publish checksums and SBOM
- Installer places binary and default config

## Local release commands

```bash
cargo build --release -p qmd-rs
./packaging/release_bundle.sh target/release/rqmd
./packaging/install.sh ~/.local/bin
```

The `release_bundle.sh` script emits `sha256sums.txt` and a minimal CycloneDX-style `sbom.json`.
