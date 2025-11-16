#!/usr/bin/env bash
# run_all.sh
# Usage:
#   bash run_all.sh                 # run all themes
#   GRADIO_URL=http://127.0.0.1:7860 INSTR_ROOT=instructions/4x3 SCENES_ROOT=scenes bash run_all.sh
#   bash run_all.sh "city" "ancient rome"   # subset (quotes needed for names with spaces)

set -euo pipefail

# Allow overrides via environment variables
GRADIO_URL="${GRADIO_URL:-http://127.0.0.1:7860}"
INSTR_ROOT="${INSTR_ROOT:-instructions/4x3}"
SCENES_ROOT="${SCENES_ROOT:-scenes}"

# Full theme list (space-containing entries must be quoted)
declare -a THEMES_DEFAULT=(
  "city"
  "medieval"
  "desert"
  "cyberpunk"
  "ancient rome"
  "minecraft"
  "forest"
  "ocean"
  "winter"
  "lego"
  "park"
  "amusement park"
  "airport"
  "college"
  "room"
)

# If args given, use them as the theme subset; else use the full list.
if (( "$#" > 0 )); then
  THEMES=("$@")
else
  THEMES=("${THEMES_DEFAULT[@]}")
fi

for theme in "${THEMES[@]}"; do
  slug="${theme// /_}"                           # "ancient rome" -> "ancient_rome"
  inst="${INSTR_ROOT}/${slug}.json"
  prefix="${SCENES_ROOT}/${slug}"

  if [[ ! -f "$inst" ]]; then
    echo "⚠️  Missing instructions file: $inst — skipping \"$theme\"."
    continue
  fi

  echo "==> Running: ${theme}"
  echo "    instructions: $inst"
  echo "    prefix      : $prefix"
  echo "    gradio_url  : $GRADIO_URL"

  python run_pipeline.py \
    --instructions "$inst" \
    --prefix "$prefix" \
    --gradio_url="$GRADIO_URL"

  python blend_gaussians.py \
    --compute_rescaled \
    --stitch_images \
    --stitch_slats \
    --gradio_url="$GRADIO_URL" \
    --prefix "$prefix"

  echo "✓ Done: ${theme}"
  echo
done
