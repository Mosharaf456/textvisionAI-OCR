#!/usr/bin/env bash

# ==============================================================================
# Production Startup Script for CPU-Based OCR Application
# ==============================================================================
# Usage:
#   ./start.sh                      # Starts FastAPI REST API server (default: port 8000)
#   ./start.sh api --port 8080      # Starts API on custom port
#   ./start.sh cli --help           # Invokes the CLI tool
#   ./start.sh test                 # Runs the full test suite
#   ./start.sh benchmark            # Runs CPU latency benchmark
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# CPU Threading Optimization for Intel/AMD Multi-Core Processors
export OCR_INTRA_THREADS="${OCR_INTRA_THREADS:-4}"
export OCR_INTER_THREADS="${OCR_INTER_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# 1. Virtual Environment Validation & Auto-Setup
# ------------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" || ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "⚙️  Virtual environment not detected. Initializing .venv..."
    python3 -m venv "$VENV_DIR"
    echo "📦 Installing production dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    echo "✓ Environment created successfully."
fi

source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------------------
# 2. Command Dispatcher
# ------------------------------------------------------------------------------
MODE="${1:-api}"

case "$MODE" in
    api|serve|start)
        shift || true
        echo "======================================================================"
        echo "🚀 Starting TextVision AI — Intelligent text recognition and extraction."
        echo "📍 Host: $HOST | Port: $PORT | Workers: $WORKERS"
        echo "🌐 Web Dashboard:    http://$HOST:$PORT"
        echo "📖 Interactive Docs: http://$HOST:$PORT/docs"
        echo "⚡ Hardware Optimization: ONNX Runtime CPU (AVX2/VNNI, intra=$OCR_INTRA_THREADS)"
        echo "======================================================================"
        exec uvicorn source.api.app:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS" \
            "$@"
        ;;

    cli)
        shift || true
        exec python -m source.cli.main "$@"
        ;;

    test)
        shift || true
        echo "🧪 Running full OCR test suite..."
        exec pytest -v tests/ "$@"
        ;;

    benchmark)
        shift || true
        echo "⚡ Running CPU inference benchmark..."
        exec python -m source.cli.main benchmark "$@"
        ;;

    help|--help|-h)
        echo "Usage: ./start.sh [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  api         Start the FastAPI REST API service (Default)"
        echo "  cli         Execute CLI commands (scan, read, invoice, id-card, alpr)"
        echo "  test        Run automated test suite via pytest"
        echo "  benchmark   Measure inference latency and throughput on CPU"
        echo "  help        Display this help message"
        echo ""
        echo "Environment Variables:"
        echo "  HOST        Bind host (default: 0.0.0.0)"
        echo "  PORT        Bind port (default: 8000)"
        echo "  WORKERS     Number of uvicorn worker processes (default: 1)"
        echo ""
        echo "Examples:"
        echo "  ./start.sh                                # Start API on port 8000"
        echo "  PORT=9000 ./start.sh                      # Start API on port 9000"
        echo "  ./start.sh cli scan doc.png --pdf out.pdf # Scan document to Searchable PDF"
        echo "  ./start.sh cli alpr frame.jpg             # Run License Plate Recognition"
        echo "  ./start.sh test                           # Run test suite"
        exit 0
        ;;

    *)
        # If unknown command, pass through to CLI
        exec python -m source.cli.main "$@"
        ;;
esac
