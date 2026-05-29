#!/usr/bin/env python3
"""
network_comparison_unified.py
=============================

ONE script to compute every AI-vs-human (and baseline-vs-human) network
comparison metric used in the paper, under a single consistent convention:
the UNION roster with zero-imputation.

WHY UNION ROSTER:
  Every node appearing in either network is in play. An edge incident to a
  node present on only one side is a genuine disagreement (false positive or
  false negative), not a silently-deleted non-comparison. This is the only
  convention under which roster expansion (e.g. a model inventing characters,
  or splitting a collective into individuals) is correctly penalized.

WHAT IT COMPUTES, per (story, comparison):
  Edge metrics
    - union_F1_collapsed   (Approach 3): multiplex collapsed to one edge/pair.
                            "Did they find the same relationships?"
    - union_F1_multiplex   (Approach 2): each (pair, tie_type) counted.
                            "Did they find AND classify the same relationships?"
    - per-layer F1         for each of the 5 tie types (union roster).
  Roster metric
    - roster_jaccard       |shared nodes| / |union nodes|
  Structural metrics (both on the UNION roster, 0-imputed)
    - degree_spearman      rank correlation of degree across union nodes
    - qap_correlation      Pearson correlation of the two flattened adjacency
                            matrices over union nodes, with a QAP permutation
                            p-value (node-label permutation, default 2000 perms)
  Network descriptives
    - n_model, n_human, edges_model, edges_human (collapsed union edges)
    - density_model, density_human, components_model, components_human

USAGE:
  python network_comparison_unified.py --dir /path/to/networks
  python network_comparison_unified.py --dir ./data --perms 5000

FILE NAMING EXPECTED (in --dir):
  edges_human_<story>.csv     (required for any comparison)
  edges_ai_<story>.csv        (optional: multi-agent vs human)
  edges_baseline_<story>.csv  (optional: baseline vs human)
  nodes_human_<story>.csv     (optional: lets isolates count in the roster)
  nodes_ai_<story>.csv        (optional)
  nodes_baseline_<story>.csv  (optional)

  Edge CSVs need columns: source, target, tie_type   (or 'relation')
  Node CSVs need column:   id
  If a nodes file is present, its ids are unioned into the roster so that
  isolate characters (no edges) still count. If absent, the roster is derived
  from the edge endpoints only.

OUTPUTS (written into --dir, or --out if given):
  comparison_edge_metrics.csv      one row per (story, comparison) edge summary
  comparison_per_layer.csv         one row per (story, comparison, tie_type)
  comparison_structural.csv        one row per (story, comparison) structural
  comparison_full_long.csv         everything in tidy long format
  Also prints readable summary tables to stdout.

No dependencies beyond pandas, numpy, networkx, scipy.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

TIES = ["family", "friendship", "romantic", "professional", "adversarial"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_edges(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "tie_type" not in df.columns and "relation" in df.columns:
        df = df.rename(columns={"relation": "tie_type"})
    if "tie_type" not in df.columns:
        raise ValueError(f"{path} has no 'tie_type' or 'relation' column")
    df["tie_type"] = df["tie_type"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["pair"] = df.apply(lambda r: tuple(sorted([r["source"], r["target"]])), axis=1)
    # drop accidental self-ties
    df = df[df["source"] != df["target"]].copy()
    return df


def load_nodes(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    col = "id" if "id" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())


def roster(edges_df, nodes_set):
    """Full node set: edge endpoints plus any declared nodes (for isolates)."""
    r = set(edges_df["source"]) | set(edges_df["target"])
    if nodes_set:
        r |= nodes_set
    return r


# ---------------------------------------------------------------------------
# Edge metric helpers
# ---------------------------------------------------------------------------

def prf(model_set, human_set):
    tp = len(model_set & human_set)
    fp = len(model_set - human_set)
    fn = len(human_set - model_set)
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else float("nan")
    return tp, fp, fn, p, r, f1


def collapsed_pairs(df):
    return set(df["pair"])


def multiplex_pairs(df):
    return set(zip(df["pair"], df["tie_type"]))


def layer_pairs(df, tt):
    return set(df[df["tie_type"] == tt]["pair"])


# ---------------------------------------------------------------------------
# Structural metric helpers (union roster, 0-imputation)
# ---------------------------------------------------------------------------

def build_graph(edges_df, node_set):
    g = nx.Graph()
    g.add_nodes_from(node_set)
    for _, row in edges_df.iterrows():
        g.add_edge(row["source"], row["target"])  # collapsed/undirected
    return g


def aligned_adjacency(g, node_order):
    n = len(node_order)
    idx = {name: i for i, name in enumerate(node_order)}
    M = np.zeros((n, n), dtype=int)
    for u, v in g.edges():
        if u in idx and v in idx:
            M[idx[u], idx[v]] = 1
            M[idx[v], idx[u]] = 1
    return M


def upper_tri(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def qap_correlation(M_model, M_human, perms, rng):
    """Pearson r between the two flattened upper-triangles, with a QAP
    permutation test (permute node labels of one matrix)."""
    a = upper_tri(M_model).astype(float)
    b = upper_tri(M_human).astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan")
    obs = np.corrcoef(a, b)[0, 1]
    n = M_model.shape[0]
    count = 0
    for _ in range(perms):
        perm = rng.permutation(n)
        Mp = M_model[np.ix_(perm, perm)]
        ap = upper_tri(Mp).astype(float)
        if ap.std() == 0:
            continue
        rp = np.corrcoef(ap, b)[0, 1]
        if abs(rp) >= abs(obs):
            count += 1
    pval = (count + 1) / (perms + 1)
    return obs, pval


def degree_spearman(g_model, g_human, node_order):
    dm = np.array([g_model.degree(n) for n in node_order], dtype=float)
    dh = np.array([g_human.degree(n) for n in node_order], dtype=float)
    if dm.std() == 0 or dh.std() == 0:
        return float("nan")
    rho, _ = spearmanr(dm, dh)
    return rho


# ---------------------------------------------------------------------------
# Per-comparison driver
# ---------------------------------------------------------------------------

def compare(story, comp_name, model_edges, human_edges,
            model_nodes, human_nodes, perms, rng):
    model_roster = roster(model_edges, model_nodes)
    human_roster = roster(human_edges, human_nodes)
    union_nodes = sorted(model_roster | human_roster)
    shared = model_roster & human_roster

    # --- roster ---
    jaccard = len(shared) / len(model_roster | human_roster) if union_nodes else float("nan")

    # --- edge metrics (union roster is implicit: sets aren't pre-filtered) ---
    m3, h3 = collapsed_pairs(model_edges), collapsed_pairs(human_edges)
    tp3, fp3, fn3, p3, r3, f3 = prf(m3, h3)

    m2, h2 = multiplex_pairs(model_edges), multiplex_pairs(human_edges)
    tp2, fp2, fn2, p2, r2, f2 = prf(m2, h2)

    # --- structural (union roster, 0-imputation) ---
    g_model = build_graph(model_edges, union_nodes)
    g_human = build_graph(human_edges, union_nodes)
    M_model = aligned_adjacency(g_model, union_nodes)
    M_human = aligned_adjacency(g_human, union_nodes)
    rho = degree_spearman(g_model, g_human, union_nodes)
    qap_r, qap_p = qap_correlation(M_model, M_human, perms, rng)

    # --- descriptives ---
    edge_row = {
        "story": story, "comparison": comp_name,
        "n_model": len(model_roster), "n_human": len(human_roster),
        "roster_jaccard": round(jaccard, 3),
        "edges_model_collapsed": len(m3), "edges_human_collapsed": len(h3),
        "union_F1_collapsed_A3": round(f3, 3) if f3 == f3 else None,
        "precision_A3": round(p3, 3) if p3 == p3 else None,
        "recall_A3": round(r3, 3) if r3 == r3 else None,
        "union_F1_multiplex_A2": round(f2, 3) if f2 == f2 else None,
        "precision_A2": round(p2, 3) if p2 == p2 else None,
        "recall_A2": round(r2, 3) if r2 == r2 else None,
    }

    struct_row = {
        "story": story, "comparison": comp_name,
        "n_union_nodes": len(union_nodes),
        "density_model": round(nx.density(g_model), 3),
        "density_human": round(nx.density(g_human), 3),
        "components_model": nx.number_connected_components(g_model),
        "components_human": nx.number_connected_components(g_human),
        "degree_spearman": round(rho, 3) if rho == rho else None,
        "qap_correlation": round(qap_r, 3) if qap_r == qap_r else None,
        "qap_p_value": round(qap_p, 4) if qap_p == qap_p else None,
    }

    # --- per-layer ---
    layer_rows = []
    for tt in TIES:
        m_set, h_set = layer_pairs(model_edges, tt), layer_pairs(human_edges, tt)
        if not m_set and not h_set:
            continue
        tp, fp, fn, p, r, f1 = prf(m_set, h_set)
        layer_rows.append({
            "story": story, "comparison": comp_name, "tie_type": tt,
            "edges_model": len(m_set), "edges_human": len(h_set),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3) if p == p else None,
            "recall": round(r, 3) if r == r else None,
            "f1": round(f1, 3) if f1 == f1 else None,
        })

    return edge_row, struct_row, layer_rows


# ---------------------------------------------------------------------------
# Discovery + main
# ---------------------------------------------------------------------------

def discover(input_dir):
    human = {p.stem.replace("edges_human_", ""): p
             for p in input_dir.glob("edges_human_*.csv")}
    ai = {p.stem.replace("edges_ai_", ""): p
          for p in input_dir.glob("edges_ai_*.csv")}
    base = {p.stem.replace("edges_baseline_", ""): p
            for p in input_dir.glob("edges_baseline_*.csv")}
    return human, ai, base


def main():
    ap = argparse.ArgumentParser(description="Unified network comparison (union roster).")
    ap.add_argument("--dir", required=True, help="Folder with the edge/node CSVs.")
    ap.add_argument("--out", default=None, help="Output folder (defaults to --dir).")
    ap.add_argument("--perms", type=int, default=2000, help="QAP permutations.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for QAP.")
    args = ap.parse_args()

    input_dir = Path(args.dir)
    out_dir = Path(args.out) if args.out else input_dir
    out_dir.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(args.seed)

    if not input_dir.exists():
        sys.exit(f"Directory not found: {input_dir.resolve()}")

    human_files, ai_files, base_files = discover(input_dir)
    if not human_files:
        sys.exit("No edges_human_*.csv files found.")

    edge_rows, struct_rows, layer_rows = [], [], []

    stories = sorted(human_files)
    print(f"Found {len(stories)} stories with human coding.\n")

    for story in stories:
        human_edges = load_edges(human_files[story])
        human_nodes = load_nodes(input_dir / f"nodes_human_{story}.csv")

        for comp_name, fdict in [("multi_agent", ai_files), ("baseline", base_files)]:
            if story not in fdict:
                continue
            model_edges = load_edges(fdict[story])
            node_key = "ai" if comp_name == "multi_agent" else "baseline"
            model_nodes = load_nodes(input_dir / f"nodes_{node_key}_{story}.csv")

            er, sr, lr = compare(story, comp_name, model_edges, human_edges,
                                 model_nodes, human_nodes, args.perms, rng)
            edge_rows.append(er)
            struct_rows.append(sr)
            layer_rows.extend(lr)
            print(f"  {story:18s} [{comp_name:11s}]  "
                  f"A3={er['union_F1_collapsed_A3']}  A2={er['union_F1_multiplex_A2']}  "
                  f"Jaccard={er['roster_jaccard']}  rho={sr['degree_spearman']}  "
                  f"QAP={sr['qap_correlation']}")

    edge_df = pd.DataFrame(edge_rows)
    struct_df = pd.DataFrame(struct_rows)
    layer_df = pd.DataFrame(layer_rows)

    edge_df.to_csv(out_dir / "comparison_edge_metrics.csv", index=False)
    struct_df.to_csv(out_dir / "comparison_structural.csv", index=False)
    layer_df.to_csv(out_dir / "comparison_per_layer.csv", index=False)

    # tidy long format combining edge + structural
    merged = edge_df.merge(struct_df, on=["story", "comparison"], how="outer")
    merged.to_csv(out_dir / "comparison_full_long.csv", index=False)

    # ---- printed summaries ----
    def summarize(df, comp):
        sub = df[df["comparison"] == comp]
        if not len(sub):
            return
        print(f"\n{'='*70}\nSUMMARY — {comp}\n{'='*70}")
        for col in ["union_F1_collapsed_A3", "union_F1_multiplex_A2",
                    "roster_jaccard"]:
            vals = sub[col].dropna()
            if len(vals):
                print(f"  {col:24s}  mean={vals.mean():.3f}  median={vals.median():.3f}")
        s2 = struct_df[struct_df["comparison"] == comp]
        for col in ["degree_spearman", "qap_correlation"]:
            vals = s2[col].dropna()
            if len(vals):
                print(f"  {col:24s}  mean={vals.mean():.3f}  median={vals.median():.3f}")

    summarize(edge_df, "multi_agent")
    summarize(edge_df, "baseline")

    if len(layer_df):
        print(f"\n{'='*70}\nPER-LAYER F1 (union roster) — multi_agent\n{'='*70}")
        ma = layer_df[layer_df["comparison"] == "multi_agent"]
        for tt in TIES:
            vals = ma[ma["tie_type"] == tt]["f1"].dropna()
            if len(vals):
                print(f"  {tt:13s} (n={len(vals):2d})  mean={vals.mean():.3f}  median={vals.median():.3f}")

    print(f"\nWrote 4 CSVs to {out_dir.resolve()}:")
    print("  comparison_edge_metrics.csv")
    print("  comparison_structural.csv")
    print("  comparison_per_layer.csv")
    print("  comparison_full_long.csv")


if __name__ == "__main__":
    main()
