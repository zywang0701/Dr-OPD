#!/usr/bin/env bash
# Render the actual README for local visual review. Requires pandoc; no GPU use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/.preview"
for language in en zh; do
  source_file=README.md
  output_file=README.preview.html
  if [ "$language" = zh ]; then source_file=README_zh.md; output_file=README_zh.preview.html; fi
  pandoc "$ROOT/$source_file" --from=gfm+tex_math_dollars --to=html5 \
    --standalone --mathjax --metadata title="Dr. OPD — README preview" \
    --css=scripts/preview.css --include-in-header="$ROOT/scripts/preview-head.html" \
    --output="$ROOT/$output_file"
done
printf 'Local previews: %s/README.preview.html and README_zh.preview.html\n' "$ROOT"
