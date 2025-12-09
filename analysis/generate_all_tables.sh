#!/bin/bash
# Batch script to generate all hypothesis testing tables

set -e  # Exit on error

echo "========================================================================"
echo "HYPOTHESIS TESTING TABLE GENERATION"
echo "========================================================================"
echo ""

# Configuration
METADATA="./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl"
GPT5_CSV="./data/model/gpt5_responses_f1-3.csv"
CLAUDE_CSV="./data/model/claude_responses_f1-3.csv"
OUTPUT_BASE="./analysis/tables"
FORMAT="txt" 

# Create output directories
mkdir -p "${OUTPUT_BASE}/gpt5"
mkdir -p "${OUTPUT_BASE}/claude"

echo "Configuration:"
echo "  Metadata: ${METADATA}"
echo "  GPT-5 CSV: ${GPT5_CSV}"
echo "  Claude CSV: ${CLAUDE_CSV}"
echo "  Output: ${OUTPUT_BASE}"
echo "  Format: ${FORMAT}"
echo ""

# Check if files exist
if [ ! -f "${METADATA}" ]; then
    echo "ERROR: Metadata file not found: ${METADATA}"
    exit 1
fi

if [ ! -f "${GPT5_CSV}" ]; then
    echo "WARNING: GPT-5 CSV not found: ${GPT5_CSV}"
    echo "Skipping GPT-5 analysis..."
    GPT5_SKIP=1
fi

if [ ! -f "${CLAUDE_CSV}" ]; then
    echo "WARNING: Claude CSV not found: ${CLAUDE_CSV}"
    echo "Skipping Claude analysis..."
    CLAUDE_SKIP=1
fi

echo "========================================================================"
echo "SINGLE MODEL TABLES"
echo "========================================================================"

# Generate tables for GPT-5-mini
if [ -z "${GPT5_SKIP}" ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Generating tables for GPT-5-mini..."
    echo "------------------------------------------------------------------------"
    python analysis/generate_hypothesis_tables.py \
        --metadata "${METADATA}" \
        --csv "${GPT5_CSV}" \
        --model-name "GPT-5-mini" \
        --output-dir "${OUTPUT_BASE}/gpt5" \
        --save-raw \
        --format "${FORMAT}"
    
    echo "GPT-5-mini tables complete"
fi

# Generate tables for Claude Sonnet 4
if [ -z "${CLAUDE_SKIP}" ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Generating tables for Claude Sonnet 4..."
    echo "------------------------------------------------------------------------"
    python analysis/generate_hypothesis_tables.py \
        --metadata "${METADATA}" \
        --csv "${CLAUDE_CSV}" \
        --model-name "Claude Sonnet 4" \
        --output-dir "${OUTPUT_BASE}/claude" \
        --save-raw \
        --format "${FORMAT}"
    
    echo "Claude Sonnet 4 tables complete"
fi

echo ""
echo "========================================================================"
echo "COMPLETE"
echo "========================================================================"
echo ""
echo "All tables generated successfully!"
echo ""
echo "Output locations:"
if [ -z "${GPT5_SKIP}" ]; then
    echo "  GPT-5-mini:  ${OUTPUT_BASE}/gpt5/"
fi
if [ -z "${CLAUDE_SKIP}" ]; then
    echo "  Claude:      ${OUTPUT_BASE}/claude/"
fi
echo ""
echo ""
echo "========================================================================"
