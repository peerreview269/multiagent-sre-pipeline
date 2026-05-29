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
- See `requirements.txt`

## Quick start

[3-4 lines: clone, install, configure endpoint, run on a sample story]

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

