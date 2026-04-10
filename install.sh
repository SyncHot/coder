#!/usr/bin/env bash
# ============================================================================
# install.sh — Dev-Assistant-OS environment setup for Ubuntu 24.04 + ROCm 6.x
# Target GPU: AMD Radeon RX 9070 XT (RDNA 4, gfx1201)
# ============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# 1. System prerequisites
# ---------------------------------------------------------------------------
info "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake pkg-config \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl wget \
    libffi-dev libssl-dev \
    rocm-hip-runtime rocm-hip-sdk 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. ROCm 6.x installation (if not already installed)
# ---------------------------------------------------------------------------
if ! command -v rocminfo &>/dev/null; then
    info "Installing ROCm 6.x repository..."

    # Add AMD GPG key and repository
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
        sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

    ROCM_VERSION="6.3.1"
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
        https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list

    # Pin priority
    cat <<EOF | sudo tee /etc/apt/preferences.d/rocm-pin-600
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        rocm-hip-runtime \
        rocm-hip-sdk \
        rocm-dev \
        rocm-libs \
        rocm-smi-lib \
        rocminfo \
        hipcc

    # Add current user to video and render groups
    sudo usermod -aG video,render "$USER"
    info "ROCm ${ROCM_VERSION} installed. You may need to log out and back in."
else
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    info "ROCm already installed: ${ROCM_VER}"
fi

# ---------------------------------------------------------------------------
# 3. Verify ROCm sees the GPU
# ---------------------------------------------------------------------------
info "Checking GPU visibility..."
if command -v rocminfo &>/dev/null; then
    GPU_NAME=$(rocminfo 2>/dev/null | grep -m1 "Marketing Name" | sed 's/.*: *//' || true)
    if [[ -n "$GPU_NAME" ]]; then
        info "Detected GPU: ${GPU_NAME}"
    else
        warn "rocminfo did not detect a GPU. Check drivers."
    fi
fi

# ---------------------------------------------------------------------------
# 4. Python virtual environment
# ---------------------------------------------------------------------------
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python 3.12 virtual environment..."
    python3.12 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel -q

# ---------------------------------------------------------------------------
# 5. Install llama-cpp-python with ROCm/HIP backend
# ---------------------------------------------------------------------------
info "Building llama-cpp-python with ROCm (HIP) support..."
# RX 9070 XT = RDNA 4 = gfx1201
export CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201"
export FORCE_CMAKE=1

pip install llama-cpp-python --force-reinstall --no-cache-dir -q 2>&1 | tail -5

# ---------------------------------------------------------------------------
# 6. Install codator and its dependencies
# ---------------------------------------------------------------------------
info "Installing codator in editable mode..."
pip install -e ".[dev]" -q

# ---------------------------------------------------------------------------
# 7. tree-sitter language grammars (for project indexing)
# ---------------------------------------------------------------------------
info "Pre-building tree-sitter grammars..."
python3 -c "
import tree_sitter_languages
print('tree-sitter grammars OK')
" 2>/dev/null || warn "tree-sitter-languages not fully available, indexing will use fallback"

# ---------------------------------------------------------------------------
# 8. Environment variables reminder
# ---------------------------------------------------------------------------
cat <<'BANNER'

╔══════════════════════════════════════════════════════════════╗
║         Dev-Assistant-OS (codator) — Setup Complete         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Activate environment:  source .venv/bin/activate            ║
║  Run CLI:               codator                              ║
║  Run web dashboard:     codator --web                        ║
║                                                              ║
║  Optional env vars:                                          ║
║    export ANTHROPIC_API_KEY="sk-ant-..."                     ║
║    export OPENAI_API_KEY="sk-..."                            ║
║                                                              ║
║  GPU target: gfx1201 (RX 9070 XT / RDNA 4)                  ║
║  ROCm docs: https://rocm.docs.amd.com                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
BANNER

info "Done! Run: source .venv/bin/activate && codator"
