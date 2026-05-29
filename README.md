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

## Adapting to your own texts and typology

[short section on what to edit: tie definitions, roster rules,
agent prompts, chunk size]

## Reproducibility

[note on the model used, hardware, seeds, etc.]
