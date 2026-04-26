#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure system tkinter is present (Homebrew Python lacks _tkinter)
echo "Installing system tkinter..."
sudo apt-get install -y python3-tk python3-full

# Deactivate any active venv
deactivate 2>/dev/null || true

# Recreate venv using the system Python so _tkinter is available
rm -rf venv
/usr/bin/python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Run the app with: ./run.sh"
