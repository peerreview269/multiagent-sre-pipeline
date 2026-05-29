"""
Multi-agent social relation extraction pipeline — entry point.

Coordinates six specialized agents (Supervisor, Roster, Entity Resolution,
Tie, Critic, Steward) through a ReAct loop to extract multiplex social
networks from narrative text. Built to run against any GGUF model served
via llama.cpp's HTTP API.

Usage:
    python run.py --story path/to/story.txt --output outputs/result.json

Configuration:
    --model PATH           Path to the local GGUF model (default: ./models/...)
    --chunk-words N        Words per chunk (default: 1400)
    --chunk-overlap N      Word overlap between chunks (default: 180)
    --max-steps N          Maximum Supervisor steps before forced finalise (default: 36)
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_llm_server import LocalLLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompts import (
    SUPERVISOR_SYSTEM,
    ROSTER_SYSTEM,
    ENTITY_RESOLUTION_SYSTEM,
    TIE_SYSTEM,
    CRITIC_SYSTEM,
    STEWARD_SYSTEM,
)

console = Console()  # replaced per-run in main() with tee'd version

TIE_TYPES = ["family", "friendship", "romantic", "professional", "adversarial"]
DEFAULT_MODEL = "./models/gemma-4-31B-it-Q5_K_M.gguf"
DEFAULT_STORY = "Dummy_Story.txt"


def build_chat(system: str, user: str, thinking: bool = False) -> str:
    think_prefix = "<|think|>\n" if thinking else ""
    return (
        f"<bos><|turn>system\n{system}<turn|>\n"
        f"<|turn>user\n{user}<turn|>\n"
        f"<|turn>model\n{think_prefix}"
    )


def parse_json(text: str, fallback: Any = None) -> Any:
    """
    Robust JSON extraction from LLM output.

    Handles the various ways local LLMs corrupt JSON: markdown fences,
    thinking-block tags, trailing commas, and truncated arrays. Tries
    progressively more permissive recovery strategies before giving up.
    """
    if fallback is None:
        fallback = {}

    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<channel\|>", "", text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_char in ["[", "{"]:
        pos = text.find(start_char)
        if pos == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            return obj
        except json.JSONDecodeError:
            pass

    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    for start_char in ["[", "{"]:
        pos = fixed.find(start_char)
        if pos == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(fixed[pos:])
            return obj
        except json.JSONDecodeError:
            pass

    array_start = text.find("[")
    if array_start != -1:
        partial = text[array_start:]
        objects = []
        pos = 0
        while pos < len(partial):
            obj_start = partial.find("{", pos)
            if obj_start == -1:
                break
            try:
                obj, end = decoder.raw_decode(partial[obj_start:])
                if isinstance(obj, dict):
                    objects.append(obj)
                pos = obj_start + end
            except json.JSONDecodeError:
                pos = obj_start + 1
        if objects:
            console.log(f"[yellow]parse_json: recovered {len(objects)} object(s) from truncated array[/yellow]")
            return objects

    return fallback


@dataclass
class StoryChunk:
    index: int
    text: str
    word_start: int
    word_end: int

    def summary(self) -> dict:
        words = self.text.split()
        preview = " ".join(words[:22])
        if len(words) > 22:
            preview += " ..."
        return {
            "index": self.index,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "preview": preview,
        }


@dataclass
class GraphState:
    story: str
    chunks: list[StoryChunk]
    characters: list[dict] = field(default_factory=list)
    candidate_relationships: list[dict] = field(default_factory=list)
    critiques: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    entity_resolution: dict = field(default_factory=dict)
    attempted_tie_types: set[str] = field(default_factory=set)
    pending_escalations: list[dict] = field(default_factory=list)
    chunk_notes: list[dict] = field(default_factory=list)

    def character_ids(self) -> set[str]:
        return {c["id"] for c in self.characters if c.get("id")}

    def coverage_summary(self) -> dict:
        done = sorted(self.attempted_tie_types)
        missing = [t for t in TIE_TYPES if t not in self.attempted_tie_types]
        counts = {}
        chunk_coverage = {t: [] for t in TIE_TYPES}
        for r in self.candidate_relationships:
            tt = r.get("tie_type", "unknown")
            counts[tt] = counts.get(tt, 0) + 1
            for idx in r.get("chunk_indexes", []):
                if tt in chunk_coverage and idx not in chunk_coverage[tt]:
                    chunk_coverage[tt].append(idx)
        for tt in chunk_coverage:
            chunk_coverage[tt] = sorted(chunk_coverage[tt])
        return {
            "attempted": done,
            "missing": missing,
            "counts_by_type": counts,
            "chunk_coverage_by_type": chunk_coverage,
        }


def chunk_story(story: str, chunk_words: int = 1400, overlap: int = 180) -> list[StoryChunk]:
    words = story.split()
    if not words:
        return [StoryChunk(index=0, text="", word_start=0, word_end=0)]
    if chunk_words <= overlap:
        raise ValueError("chunk_words must be larger than overlap")
    chunks: list[StoryChunk] = []
    step = chunk_words - overlap
    idx = 0
    start = 0
    total = len(words)
    while start < total:
        end = min(total, start + chunk_words)
        text = " ".join(words[start:end])
        chunks.append(StoryChunk(index=idx, text=text, word_start=start, word_end=end))
        if end >= total:
            break
        start += step
        idx += 1
    return chunks


def normalize_chunk_indexes(chunk_indexes: list[int] | None, chunks: list[StoryChunk]) -> list[int]:
    if not chunk_indexes:
        return [c.index for c in chunks]
    valid = {c.index for c in chunks}
    return [idx for idx in chunk_indexes if idx in valid]


def build_chunk_view(chunks: list[StoryChunk], chunk_indexes: list[int] | None = None) -> str:
    selected = normalize_chunk_indexes(chunk_indexes, chunks)
    parts = []
    by_index = {c.index: c for c in chunks}
    for idx in selected:
        c = by_index[idx]
        parts.append(f"[CHUNK {c.index} | words {c.word_start}-{c.word_end}]\n{c.text}")
    return "\n\n".join(parts)


def short_chunk_digest(chunks: list[StoryChunk], max_chunks: int = 12) -> list[dict]:
    return [c.summary() for c in chunks][:max_chunks]


def _condense_text(text: str, max_len: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_len else flat[: max_len - 3] + "..."


def build_supervisor_history_context(history: list[str], keep_last: int = 6) -> str:
    if not history:
        return "(none)"
    if len(history) <= keep_last:
        return "\n\n".join(history)
    older = history[:-keep_last]
    recent = history[-keep_last:]
    compressed = []
    for idx, entry in enumerate(older, start=1):
        tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', entry)
        tool = tool_match.group(1) if tool_match else "unknown_tool"
        obs = entry.split("OBSERVATION:\n", 1)[-1]
        compressed.append(f"{idx}. tool={tool} | observation={_condense_text(obs, 180)}")
    return (
        "Older step summary:\n"
        + "\n".join(compressed)
        + "\n\nRecent detailed steps:\n"
        + "\n\n".join(recent)
    )


def _normalise_confidence(value: Any) -> str:
    if isinstance(value, (float, int)):
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"
    return {"high": "high", "medium": "medium", "low": "low",
            "h": "high", "m": "medium", "l": "low"}.get(
        str(value).strip().lower(), "medium")


def _normalise_strength(value: Any) -> str:
    return {"strong": "strong", "moderate": "moderate", "weak": "weak",
            "high": "strong", "medium": "moderate", "low": "weak"}.get(
        str(value).strip().lower(), "moderate")


class BaseAgent:
    def __init__(self, llm: LocalLLM):
        self.llm = llm


class RosterAgent(BaseAgent):
    def run(self, chunks: list[StoryChunk], chunk_indexes: list[int] | None = None) -> list[dict]:
        selected = normalize_chunk_indexes(chunk_indexes, chunks)
        story_view = build_chunk_view(chunks, selected)
        prompt = build_chat(
            ROSTER_SYSTEM,
            f"""List every socially meaningful character in the text below.
Return ONLY a JSON array. No prose. No markdown fences. Start your response with [ and end with ].

Each object must have:
- id:      snake_case identifier (e.g. "clara_mendoza")
- name:    full display name
- aliases: list of other names used (may be empty)
- role:    one sentence
- type:    "PERSON"

TEXT:
{story_view}""",
        )
        raw = self.llm.call(prompt, max_tokens=8000)
        console.log(f"[dim]RosterAgent raw ({len(raw)} chars): {raw[:200]}[/dim]")
        chars = parse_json(raw, [])

        if not chars and len(selected) > 1:
            console.log("[yellow]RosterAgent: multi-chunk call returned empty — falling back to per-chunk extraction.[/yellow]")
            chars = []
            seen_fallback: set[str] = set()
            for single_idx in selected:
                known_block = ""
                if chars:
                    known_lines = "\n".join(
                        f'- {c["id"]}: {c["name"]} | aliases: {", ".join(c.get("aliases", []))}'
                        for c in chars
                    )
                    known_block = (
                        f"\nAlready-known characters (reuse these IDs if the same person appears):\n"
                        f"{known_lines}\n"
                    )
                single_prompt = build_chat(
                    ROSTER_SYSTEM,
                    f"""List every socially meaningful character in the text below.
Return ONLY a JSON array. No prose. No markdown fences. Start with [ and end with ].

Each object: id (snake_case), name, aliases (list), role (one sentence), type ("PERSON").
{known_block}
TEXT:
{build_chunk_view(chunks, [single_idx])}""",
                )
                raw_s = self.llm.call(single_prompt, max_tokens=8000)
                console.log(f"[dim]RosterAgent chunk {single_idx} raw ({len(raw_s)} chars): {raw_s[:200]}[/dim]")
                for c in parse_json(raw_s, []):
                    if isinstance(c, dict) and c.get("id") and c["id"] not in seen_fallback:
                        chars.append(c)
                        seen_fallback.add(c["id"])

        if not chars:
            console.log("[yellow]RosterAgent: still empty — retrying with minimal prompt.[/yellow]")
            minimal_prompt = build_chat(
                "Return only valid JSON. No explanations.",
                f"""Extract character names from this text as a JSON array.
Format: [{{"id":"snake_case","name":"Full Name","aliases":[],"role":"one sentence","type":"PERSON"}}]
Respond with ONLY the JSON array, nothing else.

TEXT:
{story_view[:2000]}""",
            )
            raw = self.llm.call(minimal_prompt, max_tokens=8000)
            console.log(f"[dim]RosterAgent minimal raw ({len(raw)} chars): {raw[:200]}[/dim]")
            chars = parse_json(raw, [])

        clean, seen = [], set()
        for c in chars:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if not cid or cid in seen:
                continue
            c.setdefault("aliases", [])
            c.setdefault("role", "")
            c.setdefault("type", "PERSON")
            clean.append(c)
            seen.add(cid)
        return clean


class EntityResolutionAgent(BaseAgent):
    def run(self, story_view: str, characters: list[dict]) -> dict:
        alias_hints = []
        name_to_ids: dict[str, list[str]] = {}
        for c in characters:
            cid = c.get("id", "")
            cname = c.get("name", "").strip().lower()
            if cname:
                name_to_ids.setdefault(cname, []).append(cid)
            for a in c.get("aliases", []):
                aname = a.strip().lower()
                if aname:
                    name_to_ids.setdefault(aname, []).append(cid)
        for name, ids in name_to_ids.items():
            unique_ids = list(dict.fromkeys(ids))
            if len(unique_ids) > 1:
                alias_hints.append(f"  '{name}' matches ids: {unique_ids} — likely duplicates")

        hints_block = (
            "Pre-detected alias overlaps (strong merge candidates):\n" + "\n".join(alias_hints)
            if alias_hints else "No automatic alias overlaps detected."
        )

        prompt = build_chat(
            ENTITY_RESOLUTION_SYSTEM,
            f"""Review this roster for duplicate or overlapping characters.

{hints_block}

Roster:
{json.dumps(characters, indent=2)}

Return a JSON object with:
- canonical_characters: cleaned list (same schema as input)
- mapping: list of {{from_id, to_id, decision, reason}}
  decision is one of "merge" | "keep_separate" | "rename"
- notes: list of short analyst notes

Rules:
- Characters listed under "Pre-detected alias overlaps" MUST be merged unless
  you have a specific textual reason they are different people.
- When merging, keep the most descriptive id (full name preferred over nickname).
- Combine aliases from both entries in the merged result.
- Do not invent characters.
- If no changes are needed, return original roster and empty mapping.

TEXT:
{story_view}""",
            thinking=True,
        )
        raw = self.llm.call_thinking(prompt, max_tokens=8000)
        result = parse_json(raw, {})
        if isinstance(result, list):
            result = {"canonical_characters": result, "mapping": [], "notes": []}
        canonical = result.get("canonical_characters", characters) or characters
        mapping = result.get("mapping", [])
        notes = result.get("notes", [])

        clean_chars, seen = [], set()
        for c in canonical:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if not cid or cid in seen:
                continue
            aliases = [a.strip() for a in c.get("aliases", []) if isinstance(a, str) and a.strip()]
            c["aliases"] = list(dict.fromkeys(aliases))
            c.setdefault("role", "")
            c.setdefault("type", "PERSON")
            clean_chars.append(c)
            seen.add(cid)

        clean_mapping = [
            {**m, "decision": m.get("decision", "keep_separate"), "reason": m.get("reason", "")}
            for m in mapping
            if isinstance(m, dict) and m.get("from_id") and m.get("to_id")
        ]
        return {
            "canonical_characters": clean_chars or characters,
            "mapping": clean_mapping,
            "notes": [n for n in notes if isinstance(n, str)],
        }

    @staticmethod
    def build_alias_id_map(characters: list[dict]) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for c in sorted(characters, key=lambda x: len(x.get("id", "")), reverse=True):
            cid = c.get("id", "")
            cname = c.get("name", "").strip().lower()
            if cname:
                alias_map.setdefault(cname, cid)
                alias_map.setdefault(cname.replace(" ", "_"), cid)
            for a in c.get("aliases", []):
                akey = a.strip().lower()
                if akey:
                    alias_map.setdefault(akey, cid)
                    alias_map.setdefault(akey.replace(" ", "_"), cid)
            alias_map.setdefault(cid.lower(), cid)
        return alias_map

    @staticmethod
    def apply_mapping(characters: list[dict], relationships: list[dict], resolution: dict) -> tuple[list[dict], list[dict]]:
        id_map = {
            item["from_id"]: item["to_id"]
            for item in resolution.get("mapping", [])
            if item.get("decision") in {"merge", "rename"}
        }
        alias_map = EntityResolutionAgent.build_alias_id_map(
            resolution.get("canonical_characters", characters) or characters
        )
        canonical = resolution.get("canonical_characters", characters) or characters
        valid_ids = {c["id"] for c in canonical if c.get("id")}

        def resolve_id(raw_id: str) -> str:
            if raw_id in id_map:
                return id_map[raw_id]
            if raw_id in valid_ids:
                return raw_id
            return alias_map.get(raw_id.lower(), raw_id)

        normalized = []
        for rel in relationships:
            r = dict(rel)
            r["source"] = resolve_id(r.get("source", ""))
            r["target"] = resolve_id(r.get("target", ""))
            if r["source"] in valid_ids and r["target"] in valid_ids:
                normalized.append(r)
        return canonical, normalized


class TieAgent(BaseAgent):
    def run(self, chunks: list[StoryChunk], characters: list[dict], tie_type: str,
            hints: list[dict] | None = None, chunk_indexes: list[int] | None = None) -> dict:
        selected = normalize_chunk_indexes(chunk_indexes, chunks)
        char_block = "\n".join(
            f'- {c["id"]}: {c["name"]} | aliases={", ".join(c.get("aliases", []))} | role={c.get("role", "")}'
            for c in characters
        )
        prompt = build_chat(
            TIE_SYSTEM,
            f"""Extract only {tie_type} ties from the selected text.

Known characters:
{char_block}

Selected chunk indexes: {selected}

Critic hints to investigate (may be empty):
{json.dumps(hints or [], indent=2)}

Return a JSON object with:
- relationships: array of relationship objects, each with:
    source, target, tie_type, sub_type, direction, strength, confidence,
    evidence, evidence_spans, temporal_note
- escalation: null, or a typed escalation object matching the system schema

TEXT:
{build_chunk_view(chunks, selected)}""",
            thinking=True,
        )
        raw = self.llm.call_thinking(prompt, max_tokens=8000)
        result = parse_json(raw, {})
        if isinstance(result, list):
            raw_rels = result
            escalation = None
        else:
            raw_rels = result.get("relationships", [])
            escalation = result.get("escalation")

        valid_ids = {c["id"] for c in characters}
        clean = []
        for r in raw_rels:
            if not isinstance(r, dict):
                continue
            if r.get("source") not in valid_ids or r.get("target") not in valid_ids:
                continue
            if r.get("source") == r.get("target"):
                continue
            spans = []
            for sp in r.get("evidence_spans", []):
                if not isinstance(sp, dict):
                    continue
                idx = sp.get("chunk_index")
                if idx in selected:
                    spans.append({
                        "chunk_index": idx,
                        "text_snippet": sp.get("text_snippet", ""),
                        "reason": sp.get("reason", ""),
                    })
            if not spans:
                spans = [{"chunk_index": idx, "text_snippet": "", "reason": "selected context"} for idx in selected[:1]]
            chunk_ids = sorted({sp["chunk_index"] for sp in spans})
            rel = {
                "source": r.get("source"),
                "target": r.get("target"),
                "tie_type": tie_type,
                "sub_type": r.get("sub_type", ""),
                "direction": r.get("direction", "undirected"),
                "strength": _normalise_strength(r.get("strength", "moderate")),
                "confidence": _normalise_confidence(r.get("confidence", "medium")),
                "evidence": r.get("evidence", ""),
                "evidence_spans": spans,
                "chunk_indexes": chunk_ids,
                "temporal_note": r.get("temporal_note", "") or "",
            }
            clean.append(rel)

        escalation = self._validate_escalation(escalation, selected, tie_type)
        return {"relationships": clean, "escalation": escalation}

    _TOOL_VALID_ARGS: dict[str, set[str]] = {
        "run_roster":            {"chunk_indexes"},
        "run_entity_resolution": {"chunk_indexes"},
        "run_tie_extraction":    {"tie_type", "hints", "chunk_indexes"},
        "inspect_chunks":        {"chunk_indexes"},
    }

    @staticmethod
    def _validate_escalation(escalation: Any, selected_chunks: list[int], tie_type: str) -> dict | None:
        if not isinstance(escalation, dict):
            return None
        ev_spans = []
        for sp in escalation.get("evidence_spans", []):
            if not isinstance(sp, dict):
                continue
            idx = sp.get("chunk_index")
            if idx in selected_chunks:
                ev_spans.append({
                    "chunk_index": idx,
                    "text_snippet": sp.get("text_snippet", ""),
                    "reason": sp.get("reason", ""),
                })
        recommended_tool = escalation.get("recommended_tool")
        if recommended_tool not in {"run_roster", "run_entity_resolution", "run_tie_extraction", "inspect_chunks"}:
            recommended_tool = "inspect_chunks"
        raw_args = escalation.get("recommended_args", {}) if isinstance(escalation.get("recommended_args"), dict) else {}
        valid_keys = TieAgent._TOOL_VALID_ARGS.get(recommended_tool, set())
        recommended_args = {k: v for k, v in raw_args.items() if k in valid_keys}
        if recommended_tool in {"inspect_chunks", "run_roster", "run_tie_extraction"} and "chunk_indexes" not in recommended_args and selected_chunks:
            recommended_args["chunk_indexes"] = selected_chunks
        if recommended_tool == "run_tie_extraction" and "tie_type" not in recommended_args:
            recommended_args["tie_type"] = tie_type
        return {
            "type": escalation.get("type", "passage_recheck"),
            "priority": escalation.get("priority", "medium"),
            "description": escalation.get("description", ""),
            "affected_entities": escalation.get("affected_entities", []),
            "recommended_tool": recommended_tool,
            "recommended_args": recommended_args,
            "suggested_action": escalation.get("suggested_action", ""),
            "evidence_spans": ev_spans,
            "raised_by": f"TieAgent:{tie_type}",
        }


class CriticAgent(BaseAgent):
    def run(self, story_view: str, chunk_digest: list[dict], characters: list[dict],
            relationships: list[dict], entity_resolution: dict | None = None) -> dict:
        rel_summary = [
            {
                "source": r.get("source"),
                "target": r.get("target"),
                "tie_type": r.get("tie_type"),
                "sub_type": r.get("sub_type"),
                "confidence": r.get("confidence"),
                "chunk_indexes": r.get("chunk_indexes", []),
                "temporal_note": r.get("temporal_note", ""),
            }
            for r in relationships
        ]
        prompt = build_chat(
            CRITIC_SYSTEM,
            f"""Audit the current social network extraction.

Chunk digest:
{json.dumps(chunk_digest, indent=2)}

Characters:
{json.dumps([{"id": c["id"], "name": c["name"]} for c in characters], indent=2)}

Entity resolution state:
{json.dumps(entity_resolution or {}, indent=2)}

Relationships (summary):
{json.dumps(rel_summary, indent=2)}

Return a JSON object with:
- gaps: list of {{source, target, suspected_tie_type, reason, recommended_chunk_indexes}}
- low_confidence_flags: list of {{source, target, tie_type, concern, recommended_chunk_indexes}}
- contradictions: list of {{description, recommended_chunk_indexes}}
- overreach_flags: list of {{source, target, tie_type, concern, recommended_chunk_indexes}}
- entity_resolution_flags: list of {{from_id, to_id, concern, recommended_chunk_indexes}}
- recommended_rechecks: list of {{tie_type, chunk_indexes, reason}}
- notes: list of short analyst notes
- quality_verdict: "good" | "needs_work" | "poor"

TEXT:
{story_view}""",
            thinking=True,
        )
        raw = self.llm.call_thinking(prompt, max_tokens=8000)
        critique = parse_json(raw, {})
        if isinstance(critique, list):
            critique = {}
        for key in (
            "gaps", "low_confidence_flags", "contradictions",
            "overreach_flags", "entity_resolution_flags",
            "recommended_rechecks", "notes"
        ):
            critique.setdefault(key, [])
        critique.setdefault("quality_verdict", "needs_work")

        has_issues = any([
            critique["gaps"],
            critique["low_confidence_flags"],
            critique["contradictions"],
            critique["overreach_flags"],
        ])
        if not has_issues and critique["quality_verdict"] == "needs_work":
            critique["quality_verdict"] = "good"
        elif has_issues and critique["quality_verdict"] == "good":
            critique["quality_verdict"] = "needs_work"

        return critique


class StewardAgent(BaseAgent):
    @staticmethod
    def _pair_key(rel: dict) -> tuple:
        src = rel.get("source")
        tgt = rel.get("target")
        if rel.get("direction", "undirected") == "undirected":
            src, tgt = sorted([src, tgt])
        return (src, tgt)

    @staticmethod
    def _dedup_key(rel: dict) -> tuple:
        src, tgt = StewardAgent._pair_key(rel)
        return (src, tgt, rel.get("tie_type", ""), rel.get("sub_type", ""), rel.get("direction", "undirected"))

    @staticmethod
    def _conf_rank(conf: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(conf, 1)

    def _best_single(self, rels: list[dict]) -> dict:
        best = rels[0]
        for rel in rels[1:]:
            if self._conf_rank(rel.get("confidence", "medium")) > self._conf_rank(best.get("confidence", "medium")):
                best = rel
            elif len(rel.get("evidence", "")) > len(best.get("evidence", "")):
                best = rel
            elif len(rel.get("evidence", "")) == len(best.get("evidence", "")):
                # Prefer earlier chunk indexes for cleaner introductory evidence
                rel_min_chunk = min(rel.get("chunk_indexes", [999]))
                best_min_chunk = min(best.get("chunk_indexes", [999]))
                if rel_min_chunk < best_min_chunk:
                    best = rel
        return best

    def _conflict_groups(self, relationships: list[dict]) -> list[dict]:
        grouped: dict[tuple, list[dict]] = {}
        for rel in relationships:
            src, tgt = self._pair_key(rel)
            key = (src, tgt, rel.get("tie_type", ""))
            grouped.setdefault(key, []).append(rel)
        conflicts = []
        for (src, tgt, tt), rels in grouped.items():
            sub_labels = {r.get("sub_type", "") for r in rels}
            if len(sub_labels) > 1:
                conflicts.append({"pair": (src, tgt), "tie_type": tt, "candidates": rels})
        return conflicts

    def adjudicate_conflicts(self, conflicts: list[dict]) -> dict:
        if not conflicts:
            return {"decisions": [], "notes": []}
        compact = []
        for item in conflicts[:12]:
            pair = item["pair"]
            compact.append({
                "pair": list(pair),
                "tie_type": item.get("tie_type", ""),
                "candidates": [
                    {
                        "source": c.get("source"),
                        "target": c.get("target"),
                        "tie_type": c.get("tie_type"),
                        "sub_type": c.get("sub_type"),
                        "direction": c.get("direction"),
                        "confidence": c.get("confidence"),
                        "evidence": c.get("evidence", "")[:180],
                        "chunk_indexes": c.get("chunk_indexes", []),
                        "temporal_note": c.get("temporal_note", ""),
                    }
                    for c in item["candidates"]
                ]
            })
        prompt = build_chat(
            STEWARD_SYSTEM,
            f"""Adjudicate the following conflict groups.
Each group is the SAME character pair AND the SAME tie_type with competing sub_types.

Return a JSON object with:
- decisions: list of {{
    pair: [source, target],
    tie_type: "<tie_type>",
    keep_strategy: "best_single" | "keep_multiple" | "drop_all_but_highest_confidence",
    keep_labels: list of {{tie_type, sub_type}},
    note: "..."
  }}
- notes: list of short analyst notes

CONFLICT_GROUPS:
{json.dumps(compact, indent=2)}""",
        )
        raw = self.llm.call(prompt, max_tokens=8000)
        result = parse_json(raw, {})
        result.setdefault("decisions", [])
        result.setdefault("notes", [])
        return result

    def merge(self, characters: list[dict], candidate_relationships: list[dict], critiques: list[dict],
              notes: list[str], entity_resolution: dict | None = None) -> dict:
        exact_best: dict[tuple, dict] = {}
        for rel in candidate_relationships:
            key = self._dedup_key(rel)
            if key not in exact_best:
                exact_best[key] = rel
            else:
                exact_best[key] = self._best_single([exact_best[key], rel])
        relationships = list(exact_best.values())

        conflicts = self._conflict_groups(relationships)
        conflict_review = self.adjudicate_conflicts(conflicts)
        decision_map: dict[tuple, dict] = {}
        for d in conflict_review.get("decisions", []):
            if not isinstance(d, dict):
                continue
            pair = d.get("pair", [])
            tt = d.get("tie_type", "")
            if isinstance(pair, list) and len(pair) == 2:
                decision_map[(pair[0], pair[1], tt)] = d

        final_relationships = []
        grouped: dict[tuple, list[dict]] = {}
        for rel in relationships:
            src, tgt = self._pair_key(rel)
            key = (src, tgt, rel.get("tie_type", ""))
            grouped.setdefault(key, []).append(rel)

        for (src, tgt, tt), rels in grouped.items():
            decision = decision_map.get((src, tgt, tt))
            if not decision:
                final_relationships.extend(rels)
                continue
            keep_labels = {
                (x.get("tie_type"), x.get("sub_type"))
                for x in decision.get("keep_labels", [])
                if isinstance(x, dict)
            }
            strategy = decision.get("keep_strategy", "best_single")
            if strategy == "keep_multiple" and keep_labels:
                final_relationships.extend([r for r in rels if (r.get("tie_type"), r.get("sub_type")) in keep_labels])
            elif strategy == "drop_all_but_highest_confidence":
                final_relationships.append(self._best_single(rels))
            elif strategy == "best_single":
                chosen = [r for r in rels if (r.get("tie_type"), r.get("sub_type")) in keep_labels] if keep_labels else rels
                final_relationships.append(self._best_single(chosen or rels))
            else:
                final_relationships.extend(rels if len(rels) == 1 else [self._best_single(rels)])
            note = decision.get("note")
            if isinstance(note, str) and note.strip():
                notes.append(note.strip())

        connected = {r["source"] for r in final_relationships} | {r["target"] for r in final_relationships}
        isolates = [c for c in characters if c.get("id") not in connected]
        tie_breakdown: dict[str, int] = {}
        degree: dict[str, int] = {}
        tie_type_chunk_coverage: dict[str, list[int]] = {t: [] for t in TIE_TYPES}

        for rel in final_relationships:
            tt = rel.get("tie_type", "unknown")
            tie_breakdown[tt] = tie_breakdown.get(tt, 0) + 1
            degree[rel["source"]] = degree.get(rel["source"], 0) + 1
            degree[rel["target"]] = degree.get(rel["target"], 0) + 1
            for idx in rel.get("chunk_indexes", []):
                if tt in tie_type_chunk_coverage and idx not in tie_type_chunk_coverage[tt]:
                    tie_type_chunk_coverage[tt].append(idx)

        for tt in tie_type_chunk_coverage:
            tie_type_chunk_coverage[tt] = sorted(tie_type_chunk_coverage[tt])

        hub_id = max(degree, key=degree.get) if degree else None
        hub_name = next((c["name"] for c in characters if c.get("id") == hub_id), hub_id)

        return {
            "characters": characters,
            "relationships": final_relationships,
            "isolates": isolates,
            "summary": {
                "total_characters": len(characters),
                "total_relationships": len(final_relationships),
                "tie_breakdown": tie_breakdown,
                "isolate_count": len(isolates),
                "network_hub": hub_name,
                "critique_rounds": len(critiques),
                "entity_resolution_actions": len((entity_resolution or {}).get("mapping", [])),
                "conflict_groups_reviewed": len(conflicts),
                "conflict_review_batches": conflict_review.get("batches", 0),
                "tie_type_chunk_coverage": tie_type_chunk_coverage,
            },
            "analyst_notes": list(dict.fromkeys(notes + conflict_review.get("notes", []))),
            "critiques": critiques,
            "entity_resolution": entity_resolution or {},
            "steward_conflict_review": conflict_review,
        }

    def run_llm_summary(self, relationships: list[dict], characters: list[dict]) -> list[str]:
        compact = {
            "character_count": len(characters),
            "relationship_count": len(relationships),
            "tie_breakdown": {},
        }
        for r in relationships:
            tt = r.get("tie_type", "unknown")
            compact["tie_breakdown"][tt] = compact["tie_breakdown"].get(tt, 0) + 1
        prompt = build_chat(
            STEWARD_SYSTEM,
            f"""Given this extraction summary, produce a JSON array of 2-5 concise
analyst notes about ambiguity, sparsity, temporal caution, or unresolved uncertainty.

SUMMARY:
{json.dumps(compact, indent=2)}""",
        )
        raw = self.llm.call(prompt, max_tokens=8000)
        notes = parse_json(raw, [])
        return [n for n in notes if isinstance(n, str)]


class SupervisorAgent(BaseAgent):
    def __init__(self, llm: LocalLLM):
        super().__init__(llm)
        self.history: list[str] = []

    def next_action(self, state: GraphState) -> str:
        history_text = build_supervisor_history_context(self.history)
        pending_escalations = [
            {
                "type": e.get("type"),
                "priority": e.get("priority"),
                "description": e.get("description"),
                "recommended_tool": e.get("recommended_tool"),
                "recommended_args": e.get("recommended_args", {}),
            }
            for e in state.pending_escalations[:5]
        ]
        digest = {
            "characters_loaded": len(state.characters),
            "coverage": state.coverage_summary(),
            "critic_rounds": len(state.critiques),
            "last_quality_verdict": state.critiques[-1].get("quality_verdict") if state.critiques else "n/a",
            "pending_escalations": pending_escalations,
            "notes_count": len(state.notes),
            "chunk_count": len(state.chunks),
        }
        if not state.characters or len(self.history) < 3:
            digest["chunk_digest"] = short_chunk_digest(state.chunks)
        prompt = build_chat(
            SUPERVISOR_SYSTEM,
            f"""STATE:
{json.dumps(digest, indent=2)}

RECENT STEPS (do not re-analyse, just use as context):
{history_text}

Your next action:
THOUGHT: <1-2 sentences>
ACTION: {{ "tool": "<n>", "args": {{ ... }} }}""",
            thinking=True,
        )
        return self.llm.call_thinking(prompt, max_tokens=8000, stop=["OBSERVATION:", "<turn|>"])

    def record(self, supervisor_output: str, observation: str):
        clean = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", supervisor_output, flags=re.DOTALL).strip()
        clean = re.sub(r"<channel\|>", "", clean).strip()
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        self.history.append(f"{clean}\nOBSERVATION:\n{observation}")


def _parse_supervisor_action(text: str) -> dict | None:
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<channel\|>", "", text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    action_pos = text.find("ACTION:")
    search_text = text[action_pos + len("ACTION:"):] if action_pos != -1 else text
    brace_pos = search_text.find("{")
    if brace_pos == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(search_text[brace_pos:].lstrip())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class ToolDispatcher:
    def __init__(self, state: GraphState, llm: LocalLLM):
        self.state = state
        self.roster = RosterAgent(llm)
        self.er = EntityResolutionAgent(llm)
        self.tie = TieAgent(llm)
        self.critic = CriticAgent(llm)
        self.steward = StewardAgent(llm)

    def dispatch(self, tool: str, args: dict) -> tuple[str, bool]:
        s = self.state

        if tool == "inspect_chunks":
            indexes = normalize_chunk_indexes(args.get("chunk_indexes"), s.chunks)
            excerpts = []
            by_idx = {c.index: c for c in s.chunks}
            for idx in indexes[:8]:
                c = by_idx[idx]
                excerpts.append({
                    "index": idx,
                    "word_range": [c.word_start, c.word_end],
                    "excerpt": c.text[:600],
                })
            obs = json.dumps({"chunk_excerpts": excerpts}, indent=2)
            console.print(Panel(obs, title="Chunk Inspector", border_style="white"))
            return obs, False

        if tool == "run_roster":
            selected = normalize_chunk_indexes(args.get("chunk_indexes"), s.chunks)
            chars = self.roster.run(s.chunks, selected)
            existing = {c.get("id"): c for c in s.characters if c.get("id")}
            for c in chars:
                cid = c["id"]
                if cid not in existing:
                    existing[cid] = c
                else:
                    aliases = list(dict.fromkeys(existing[cid].get("aliases", []) + c.get("aliases", [])))
                    existing[cid]["aliases"] = aliases
                    if not existing[cid].get("role") and c.get("role"):
                        existing[cid]["role"] = c["role"]
                    if c.get("type") and c.get("type") != existing[cid].get("type"):
                        existing[cid]["type"] = c["type"]
            s.characters = list(existing.values())
            s.chunk_notes.append({"tool": "run_roster", "chunk_indexes": selected})
            obs = f"RosterAgent refreshed roster from chunks {selected}. Total characters now: {len(s.characters)}."
            console.print(Panel(json.dumps(s.characters, indent=2), title="RosterAgent", border_style="green"))
            return obs, False

        if tool == "run_entity_resolution":
            selected = normalize_chunk_indexes(args.get("chunk_indexes"), s.chunks)
            resolution = self.er.run(build_chunk_view(s.chunks, selected), s.characters)
            s.characters, s.candidate_relationships = self.er.apply_mapping(
                s.characters, s.candidate_relationships, resolution
            )
            s.entity_resolution = resolution
            s.notes.extend(resolution.get("notes", []))
            n_merges = sum(1 for m in resolution.get("mapping", []) if m.get("decision") == "merge")
            alias_map = self.er.build_alias_id_map(s.characters)
            valid_ids = {c["id"] for c in s.characters if c.get("id")}
            normalised = []
            for rel in s.candidate_relationships:
                r = dict(rel)
                r["source"] = alias_map.get(r.get("source", "").lower(), r.get("source", ""))
                r["target"] = alias_map.get(r.get("target", "").lower(), r.get("target", ""))
                if r["source"] in valid_ids and r["target"] in valid_ids:
                    normalised.append(r)
            s.candidate_relationships = normalised
            obs = f"EntityResolutionAgent on chunks {selected}: {n_merges} LLM merge(s), {len(s.characters)} canonical characters. Alias normalisation applied."
            console.print(Panel(json.dumps(resolution, indent=2), title="EntityResolutionAgent", border_style="blue"))
            return obs, False

        if tool == "run_tie_extraction":
            tie_type = args.get("tie_type")
            if tie_type not in TIE_TYPES:
                return f"ERROR: unknown tie_type '{tie_type}'. Choose from {TIE_TYPES}.", False
            selected = normalize_chunk_indexes(args.get("chunk_indexes"), s.chunks)
            hints = args.get("hints", [])
            result = self.tie.run(s.chunks, s.characters, tie_type, hints=hints, chunk_indexes=selected)
            rels = result["relationships"]
            escalation = result["escalation"]
            s.candidate_relationships.extend(rels)
            s.attempted_tie_types.add(tie_type)
            obs_parts = [f"TieAgent[{tie_type}] on chunks {selected}: found {len(rels)} relationship(s)."]
            if escalation:
                s.pending_escalations.append(escalation)
                obs_parts.append(
                    "\n⚠ ESCALATION raised:\n"
                    f"  type: {escalation['type']}\n"
                    f"  priority: {escalation['priority']}\n"
                    f"  description: {escalation['description']}\n"
                    f"  recommended_tool: {escalation['recommended_tool']}\n"
                    f"  recommended_args: {json.dumps(escalation['recommended_args'])}\n"
                    "Supervisor should resolve it before broadening search."
                )
                console.print(Panel(json.dumps(escalation, indent=2), title=f"ESCALATION — {tie_type}", border_style="red"))
            console.print(Panel(json.dumps(rels, indent=2), title=f"TieAgent — {tie_type}", border_style="yellow"))
            return "\n".join(obs_parts), False

        if tool == "run_critic":
            snapshot = self.steward.merge(s.characters, s.candidate_relationships, s.critiques, s.notes, s.entity_resolution)
            critique = self.critic.run(
                build_chunk_view(s.chunks),
                short_chunk_digest(s.chunks),
                s.characters,
                snapshot["relationships"],
                entity_resolution=s.entity_resolution,
            )
            s.critiques.append(critique)
            s.notes.extend([n for n in critique.get("notes", []) if isinstance(n, str)])
            verdict = critique.get("quality_verdict", "?")
            obs = (
                f"CriticAgent: verdict='{verdict}', gaps={len(critique.get('gaps', []))}, "
                f"rechecks={critique.get('recommended_rechecks', [])}, ER flags={len(critique.get('entity_resolution_flags', []))}."
            )
            console.print(Panel(json.dumps(critique, indent=2), title="CriticAgent", border_style="red"))
            return obs, False

        if tool == "run_steward":
            snapshot = self.steward.merge(s.characters, s.candidate_relationships, s.critiques, s.notes, s.entity_resolution)
            obs = (
                f"StewardAgent: {len(snapshot['relationships'])} relationship(s), {len(snapshot['isolates'])} isolate(s), "
                f"conflict_groups_reviewed={snapshot['summary']['conflict_groups_reviewed']}, batches={snapshot['summary']['conflict_review_batches']}."
            )
            console.print(Panel(json.dumps(snapshot["summary"], indent=2), title="StewardAgent", border_style="magenta"))
            return obs, False

        if tool == "retract_relationships":
            src = args.get("source")
            tgt = args.get("target")
            tt  = args.get("tie_type")
            before = len(s.candidate_relationships)
            def _matches(r: dict) -> bool:
                pair_match = (
                    (r.get("source") == src and r.get("target") == tgt) or
                    (r.get("source") == tgt and r.get("target") == src and
                     r.get("direction", "undirected") == "undirected")
                )
                type_match = (tt is None) or (r.get("tie_type") == tt)
                return pair_match and type_match
            s.candidate_relationships = [r for r in s.candidate_relationships if not _matches(r)]
            removed = before - len(s.candidate_relationships)
            obs = f"Retracted {removed} relationship(s) matching source='{src}', target='{tgt}', tie_type='{tt}'."
            console.log(f"[yellow]{obs}[/yellow]")
            return obs, False

        if tool == "resolve_escalation":
            idx = args.get("index", 0)
            action = args.get("action", "acknowledged")
            if not s.pending_escalations:
                return "No pending escalations to resolve.", False
            esc = s.pending_escalations.pop(min(idx, len(s.pending_escalations) - 1))
            base_obs = (
                f"Escalation resolved: type='{esc['type']}', priority='{esc['priority']}', "
                f"supervisor action='{action}'. Remaining escalations: {len(s.pending_escalations)}."
            )
            recommendation = ""
            if esc.get("recommended_tool"):
                recommendation = (
                    "\nRecommended next step (not auto-executed): "
                    + json.dumps({
                        "tool": esc.get("recommended_tool"),
                        "args": esc.get("recommended_args", {}),
                    })
                )
            return base_obs + recommendation, False

        if tool == "finalise":
            if not s.critiques:
                return "ERROR: run_critic at least once before finalise.", False
            if s.pending_escalations:
                return f"ERROR: cannot finalise with {len(s.pending_escalations)} pending escalation(s).", False
            result = self.steward.merge(s.characters, s.candidate_relationships, s.critiques, s.notes, s.entity_resolution)
            result["chunking"] = {
                "chunk_count": len(s.chunks),
                "chunks": [c.summary() for c in s.chunks],
            }
            result["analyst_notes"] = list(dict.fromkeys(
                result.get("analyst_notes", []) +
                self.steward.run_llm_summary(result["relationships"], result["characters"])
            ))
            s._final_result = result  # type: ignore[attr-defined]
            obs = (
                f"FINALISED. {result['summary']['total_relationships']} relationships, "
                f"{result['summary']['isolate_count']} isolate(s), chunks={len(s.chunks)}."
            )
            console.print(Panel(json.dumps(result["summary"], indent=2), title="FINAL RESULT", border_style="green"))
            return obs, True

        return f"ERROR: unknown tool '{tool}'.", False


def run(story_path: str, model_path: str, max_steps: int, chunk_words: int, chunk_overlap: int) -> dict:
    story = Path(story_path).read_text(encoding="utf-8")
    llm = LocalLLM(model_path=model_path)
    chunks = chunk_story(story, chunk_words=chunk_words, overlap=chunk_overlap)
    state = GraphState(story=story, chunks=chunks)
    supervisor = SupervisorAgent(llm)
    dispatcher = ToolDispatcher(state, llm)

    console.rule("[bold cyan]Agentic Supervisor + Typed Escalations + Chunk-Aware Rereads[/bold cyan]")

    last_action_key: str = ""
    last_action_count: int = 0

    for step in range(1, max_steps + 1):
        console.rule(f"[yellow]Step {step}/{max_steps}[/yellow]")
        supervisor_output = supervisor.next_action(state)
        console.print(Panel(supervisor_output, title="Supervisor", border_style="cyan"))
        action = _parse_supervisor_action(supervisor_output)
        if action is None:
            obs = 'ERROR: could not parse ACTION JSON. Emit ACTION: { "tool": "...", "args": { ... } }'
            console.log(f"[red]{obs}[/red]")
            supervisor.record(supervisor_output, obs)
            continue
        tool = action.get("tool", "")
        args = action.get("args", {})

        action_key = f"{tool}:{json.dumps(args, sort_keys=True)}"
        if action_key == last_action_key:
            last_action_count += 1
        else:
            last_action_key = action_key
            last_action_count = 1
        if last_action_count >= 3:
            obs = (
                f"LOOP DETECTED: '{tool}' called with the same args {last_action_count} times in a row. "
                "Choose a DIFFERENT tool or different args. Move on."
            )
            console.log(f"[bold red]{obs}[/bold red]")
            supervisor.record(supervisor_output, obs)
            last_action_count = 0
            continue

        console.print(f"[bold green]→ Tool:[/bold green] {tool}  [bold green]Args:[/bold green] {json.dumps(args)}")
        observation, is_final = dispatcher.dispatch(tool, args)
        console.print(Panel(observation, title="Observation", border_style="dim"))
        supervisor.record(supervisor_output, observation)
        if is_final:
            console.log("[bold green]Extraction complete.[/bold green]")
            break
    else:
        console.log("[yellow]Max steps reached — forcing finalise if legal.[/yellow]")
        if state.pending_escalations:
            state.pending_escalations.clear()
        if not state.critiques:
            dispatcher.dispatch("run_critic", {})
        final_obs, _ = dispatcher.dispatch("finalise", {})
        supervisor.record("(forced finalise)", final_obs)

    return getattr(state, "_final_result", {})


def pretty_relationship_table(relationships: list[dict]) -> Table:
    tie_colors = {
        "family": "green", "friendship": "cyan", "romantic": "magenta",
        "professional": "yellow", "adversarial": "red",
    }
    table = Table(title="Relationships", show_header=True, header_style="bold cyan")
    for col in ("Source", "Target", "Type", "Sub-type", "Strength", "Conf.", "Chunks", "Evidence"):
        table.add_column(col, max_width=42 if col == "Evidence" else None)
    for r in relationships:
        color = tie_colors.get(r.get("tie_type", ""), "white")
        table.add_row(
            str(r.get("source", "")),
            str(r.get("target", "")),
            f"[{color}]{r.get('tie_type', '')}[/{color}]",
            str(r.get("sub_type", "")),
            str(r.get("strength", "")),
            str(r.get("confidence", "")),
            ", ".join(str(x) for x in r.get("chunk_indexes", [])),
            str(r.get("evidence", ""))[:120],
        )
    return table


def main():
    parser = argparse.ArgumentParser(description="Multi-agent social network extractor")
    parser.add_argument("--story", default=DEFAULT_STORY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--chunk-words", type=int, default=1400)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument("--output", default="outputs/network_output.json")
    args = parser.parse_args()

    # Auto-name log file to match JSON output
    output_path = Path(args.output)
    log_path = output_path.with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tee Rich console output to both terminal and log file simultaneously
    global console
    log_file = open(log_path, "w", encoding="utf-8")
    from rich.console import Console as _Console
    console = _Console(record=True)

    _original_print = console.print
    _original_log   = console.log
    _original_rule  = console.rule

    def _tee_print(*args, **kwargs):
        _original_print(*args, **kwargs)
        kwargs.pop("highlight", None)
        _Console(file=log_file, highlight=False, markup=False).print(*args, **kwargs)

    def _tee_log(*args, **kwargs):
        _original_log(*args, **kwargs)
        kwargs.pop("highlight", None)
        _Console(file=log_file, highlight=False, markup=False).log(*args, **kwargs)

    def _tee_rule(title="", **kwargs):
        _original_rule(title, **kwargs)
        log_file.write(f"\n{'─' * 60} {title} {'─' * 60}\n\n")
        log_file.flush()

    console.print = _tee_print
    console.log   = _tee_log
    console.rule  = _tee_rule

    console.print(f"[dim]Agent log: {log_path}[/dim]")

    try:
        result = run(args.story, args.model, args.max_steps, args.chunk_words, args.chunk_overlap)

        # Save JSON FIRST — before any display that could crash
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        console.print(f"\n[bold green]Saved JSON:[/bold green] {output_path}")
        console.print(f"[bold green]Saved log: [/bold green] {log_path}")

        # Display results — crashes here won't lose the output
        console.rule("[bold magenta]FINAL GRAPH[/bold magenta]")
        console.print(Panel(json.dumps(result.get("summary", {}), indent=2), title="Summary", border_style="magenta"))
        console.print(pretty_relationship_table(result.get("relationships", [])))
        if result.get("isolates"):
            console.print(Panel("\n".join(f"  • {i['name']}" for i in result["isolates"]), title="[red]Isolates[/red]", border_style="red"))

    finally:
        log_file.close()


if __name__ == "__main__":
    main()
