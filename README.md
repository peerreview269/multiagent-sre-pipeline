# multiagent-sre-pipeline
Multi-agent pipeline for extracting multiplex social networks from document-length narrative text. Code and configuration for the methods paper currently under review.
# Multi-agent social relation extraction pipeline

Code and configuration for the multi-agent pipeline described in
"[Anonymized paper title]," currently under review.

## What this is

A document-level relation extraction system that extracts multiplex
social networks from narrative text. Six agents (Supervisor, Roster,
Entity Resolution, Tie, Critic, Steward) coordinate via a ReAct loop
to produce a multiplex edgelist with node and tie type definitions
specified by the researcher.

## Requirements

- Python 3.10+
- A locally-served LLM endpoint (we used Gemma 4 31B via llama.cpp)
- pandas>=2.0
- numpy>=1.24
- networkx>=3.0
- scipy>=1.10
- requests>=2.28
- rich>=13.0

## Quick start

# 1. Run the pipeline
python run.py --story Bartleby.txt --output outputs/bartleby_result.json

# 2. Convert the JSON to Excel + UCINET + iGraph + NetworkX
python network_to_excel.py outputs/bartleby_result.json --outdir reports/bartleby

# 3. (Optional, for paper replication only) Compare against the reference network
cp reports/bartleby/networkx/edges_nx.csv data/edges_ai_bartleby_test.csv
cp reports/bartleby/networkx/nodes_nx.csv data/nodes_ai_bartleby_test.csv
python comparison/network_comparison_unified.py --dir data/

## Repository contents

- `agents/` — system prompts for each agent
- `pipeline/` — orchestration code
- `config/` — tie typology, codebook rules, roster definition
- `corpus/` — the ten short stories (sourced from public domain
  where applicable; see `corpus/README.md` for sourcing notes)
- `comparison/` — evaluation scripts (network_comparison_unified.py)
- `data/` — extracted networks and human reference networks

## Configuring your own typology

The five tie types — family, friendship, romantic, professional, and
adversarial — are defined in `prompts.py` under `TIE_SYSTEM`. To use
your own typology, edit the definitions, sub_types, notes, and priority
rules in that constant. The Roster Agent's definition of a "socially
meaningful character" is in `ROSTER_SYSTEM` and can be similarly
adapted.

## Reproducibility

Model: Gemma 4 31B (Q5_K_M quantization)
Inference: temperature 0.7 for structured calls, 1.0 for reasoning calls
Context: 16,384 tokens
Chunk size: 700 tokens with 100-token overlap

