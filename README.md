# Multi-agent social relation extraction pipeline

Code, configuration, and evaluation data for the multi-agent pipeline
described in "[Anonymized paper title]," currently under review.

## What this is

A document-level relation extraction system that extracts multiplex social
networks from narrative text. Six specialized agents (Supervisor, Roster,
Entity Resolution, Tie, Critic, and Steward) coordinate via a ReAct loop
to produce a multiplex edgelist covering five relationship types: family,
friendship, romantic, professional, and adversarial.

The pipeline runs locally against any GGUF model served via llama.cpp's
HTTP API. The paper uses Gemma 4 31B (Q5_K_M quantization) on a single
consumer workstation.

## Requirements

- Python 3.10+
- A running llama.cpp server on `http://127.0.0.1:8080`
- See `requirements.txt` for Python packages

Start the model server before running the pipeline:

```
./llama.cpp/llama-server \
    --model ./models/gemma-4-31B-it-Q5_K_M.gguf \
    --n-gpu-layers -1 \
    --port 8080
```

## Repository contents

```
.
├── run.py                            Pipeline entry point
├── prompts.py                        System prompts for all six agents
├── local_llm_server.py               HTTP client for the llama.cpp server
├── network_to_excel.py               Convert JSON output to Excel + UCINET + iGraph + NetworkX
│
├── Bartleby.txt                      Demo input (public domain)
│
├── comparison/
│   └── network_comparison_unified.py   Evaluation script for paper replication
│
├── data/
│   ├── edges_human_*.csv             Reference (human-adjudicated) networks
│   ├── nodes_human_*.csv
│   ├── edges_ai_*.csv                Multi-agent pipeline extractions
│   ├── nodes_ai_*.csv
│   ├── edges_baseline_*.csv          Zero-shot baseline extractions
│   └── nodes_baseline_*.csv
│
├── models/                           Drop your GGUF model file here
└── outputs/                          Pipeline writes JSON + log here
```

## Demo: running the pipeline on Bartleby

Three steps. The first runs the extraction, the second converts the JSON
output to readable formats, the third (optional) compares the new
extraction against the human-adjudicated reference network.

### 1. Extract a network from a story

```
python run.py \
    --story Bartleby.txt \
    --max-steps 48 \
    --chunk-words 700 \
    --chunk-overlap 100 \
    --output outputs/bartleby_result.json \
    2>&1 | tee outputs/bartleby_result.log
```

The pipeline writes the extracted network to the specified JSON file and
a full agent-by-agent log to a parallel `.log` file. To strip terminal
color codes from the log for cleaner reading:

```
sed 's/\x1B\[[0-9;]*[mK]//g' outputs/bartleby_result.log > outputs/bartleby_result.txt
```

### 2. Convert the JSON to Excel, UCINET, iGraph, and NetworkX

```
python network_to_excel.py outputs/bartleby_result.json --outdir reports/bartleby
```

This produces:
- A formatted multisheet `.xlsx` (summary, characters, relationships,
  per-tie-type sheets, adjacency matrices, evidence detail, critic report)
- `ucinet_multiplex.txt` — UCINET DL multiplex matrix
- `edges_igraph.csv` and `nodes_igraph.csv` — iGraph-ready
- `networkx/` — GraphML, GEXF, edge/node CSVs, per-layer GraphML files

### 3. (Paper replication only) Compare against the reference network

The repo's `data/` folder includes the human-adjudicated reference
networks for all ten stories in the paper, plus the extractions reported
in §4. To reproduce the paper's quantitative comparisons:

```
python comparison/network_comparison_unified.py --dir data/ --perms 5000
```

This computes A2 (multiplex) and A3 (collapsed) F1, per-layer F1, roster
Jaccard, degree Spearman correlation, and QAP correlation with permutation
p-values, all under the union-roster convention described in §3.4 of the
paper. Output is written to four CSV files in `data/` and printed as
summary tables to stdout.

## Adapting the pipeline to your own typology

The pipeline is built to be configured for relational typologies other
than the five used in the paper. Two places to edit:

**Tie typology** — `prompts.py`, in the `TIE_SYSTEM` constant. Define
your own tie types, sub_types, notes, and priority rules. The Tie Agent
will extract one tie type per pass through the document; add as many or
as few as you need.

**Roster definition** — `prompts.py`, in the `ROSTER_SYSTEM` constant.
The current definition of a "socially meaningful character" is calibrated
for short fiction. Researchers working with, say, corporate documents or
ethnographic field notes may want to constrain the roster to named
persons and organizational entities, or expand it to include roles or
collectives unique to their texts.

The other agents (Supervisor, Entity Resolution, Critic, Steward) are
typology-agnostic and need no changes.

## Reproducibility notes

- Model: Gemma 4 31B (Q5_K_M quantization)
- Inference: temperature 0.7 for structured calls, 1.0 for reasoning calls
- Context: 16,384 tokens
- Chunk size: 700 words with 100-word overlap (the values used in the paper)
- The `--max-steps 48` cap was empirically sufficient for all ten stories
  in the corpus

The pipeline's output is non-deterministic at the LLM call level even at
fixed temperature, since the inference server does not guarantee
deterministic sampling. Reruns will produce slightly different networks
for the same input. The reference networks in `data/` are the exact
extractions reported in the paper.

## Corpus

The paper uses ten short stories spanning 1839–2011. We ship the
extracted networks for all ten (in `data/`) plus the public-domain demo
text (`Bartleby.txt`). For copyright reasons we do not redistribute the
other nine story texts. See `data/README.md` for citations and sourcing
notes.

## License

MIT (see `LICENSE`).
