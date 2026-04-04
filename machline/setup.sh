#!/usr/bin/env bash
# Setup script for MachLine fork
# Clones MachLine, installs Miniforge (for gfortran via conda-forge),
# creates a Python venv, compiles the Fortran solver, and runs examples.

set -e

MACHLINE_DIR="${1:-$HOME/MachLine}"

# 1. Clone (skip if already exists)
if [ ! -d "$MACHLINE_DIR" ]; then
    echo "Cloning MachLine..."
    git clone https://github.com/usuaero/MachLine.git "$MACHLINE_DIR"
fi

# 2. Install Miniforge + gfortran (conda-forge) if gfortran not on PATH
if ! command -v gfortran &>/dev/null; then
    echo "Installing Miniforge to get gfortran..."
    MINIFORGE_PREFIX="/opt/miniforge"
    if [ ! -d "$MINIFORGE_PREFIX" ]; then
        curl -sL -o /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        bash /tmp/miniforge.sh -b -p "$MINIFORGE_PREFIX"
    fi
    "$MINIFORGE_PREFIX/bin/conda" install -y -c conda-forge gfortran
    export PATH="$MINIFORGE_PREFIX/bin:$PATH"
fi

echo "gfortran: $(gfortran --version | head -1)"

# 3. Python virtual environment
if [ ! -d "$MACHLINE_DIR/venv" ]; then
    python3 -m venv "$MACHLINE_DIR/venv"
    "$MACHLINE_DIR/venv/bin/pip" install -q numpy matplotlib vtk
fi

# 4. Compile MachLine
cd "$MACHLINE_DIR"
# Use -std=legacy so gfortran 15+ accepts the legacy FORMAT strings in the source
gfortran -O2 -fdefault-real-8 -fopenmp -std=legacy \
    -o machline.exe \
    common/helpers.f90 common/linked_list.f90 common/math.f90 \
    common/linalg.f90 common/json.f90 common/json_xtnsn.f90 common/sort.f90 \
    src/flow.f90 src/base_geom.f90 src/panel.f90 src/mesh.f90 \
    src/stl.f90 src/vtk.f90 src/tri.f90 src/wake_strip.f90 \
    src/wake_mesh.f90 src/surface_mesh.f90 src/panel_solver.f90 src/main.f90

echo "machline.exe built: $(ls -lh machline.exe | awk '{print $5}')"
echo "Setup complete. Run examples with: ./run_examples.sh"
