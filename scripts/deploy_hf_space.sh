#!/usr/bin/env bash
# Assemble and push the Hugging Face Space.
#
# The Space needs only what the container runs -- source, app, model artifacts,
# Dockerfile, runtime requirements -- so this stages exactly those into a temp
# clone of the Space repo rather than pushing the whole project (which carries
# the DVC cache, MLflow database, and legacy files the demo never touches).
#
# Usage:
#   ./scripts/deploy_hf_space.sh <hf-username> <space-name>
#
# Prerequisites:
#   1. Create the Space at https://huggingface.co/new-space (SDK: Docker, Blank)
#   2. Create a write token at https://huggingface.co/settings/tokens
#   3. Either `pip install huggingface_hub && huggingface-cli login`,
#      or let git prompt for your username and paste the token as the password.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <hf-username> <space-name>" >&2
    exit 1
fi

HF_USER="$1"
SPACE_NAME="$2"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$(mktemp -d)"

cleanup() { rm -rf "${STAGING}"; }
trap cleanup EXIT

echo "Cloning ${SPACE_URL}"
git clone "${SPACE_URL}" "${STAGING}/space"

cd "${STAGING}/space"
git config user.name "Manoj Mareedu"
git config user.email "manoj.mareedu.pro@gmail.com"

# Clear previous contents but keep git history.
find . -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +

cp -R "${REPO_ROOT}/src" .
cp -R "${REPO_ROOT}/app" .
cp -R "${REPO_ROOT}/scripts" .
cp -R "${REPO_ROOT}/exported_model" .
cp "${REPO_ROOT}/Dockerfile" .
cp "${REPO_ROOT}/requirements.txt" .
cp "${REPO_ROOT}/huggingface/README.md" README.md

# The model artifacts are a few megabytes of binary; LFS keeps the Space repo
# healthy as the model is retrained and replaced over time.
git lfs install --local 2>/dev/null || true
git lfs track "exported_model/**/*.pkl" 2>/dev/null || true
git lfs track "exported_model/*.parquet" 2>/dev/null || true
[ -f .gitattributes ] && git add .gitattributes

git add -A
if git diff --cached --quiet; then
    echo "Nothing changed; Space is already up to date."
    exit 0
fi

git commit -m "Deploy claims cost intelligence dashboard and API"
git push

echo
echo "Pushed. The Space will build now (first build takes several minutes)."
echo "Watch it at: ${SPACE_URL}"
