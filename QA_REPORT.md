# Codator QA Report: Model Comparison for Agent Mode Bug Fixing

**Date**: 2025-07-15  
**Target**: `backend/blueprints/video_station.py` (ethos project)  
**Test Bug**: SRT→VTT subtitle conversion corrupts dialogue text  
**Server**: pluton (192.168.50.119) — AMD Ryzen 9 7900X, 16GB VRAM  

---

## Bugs Found in video_station.py

### Bug 1 — Operator Precedence (lines 1049-1050) 🔴 HIGH
```python
"tmdb_cast": r["tmdb_cast"] or "" if "tmdb_cast" in r.keys() else "",
```
Evaluates as `r["tmdb_cast"] or ("" if ... else "")` — should be `(r["tmdb_cast"] or "") if "tmdb_cast" in r.keys() else ""`

### Bug 2 — SRT→VTT conversion destroys subtitle text (line 1770) 🔴 HIGH
```python
vtt = "WEBVTT\n\n" + srt_content.replace(',', '.')
```
Replaces ALL commas with periods — corrupts dialogue. Fix: only replace commas in timestamp lines.

### Bug 3 — remove_from_library incomplete cleanup (line 1551) 🟡 MEDIUM
Misses `_BACKDROP_DIR` and `_THUMBSTRIP_DIR` when removing files.

### Bug 4 — `_hide_unlocked` memory leak 🟡 MEDIUM
Tokens never expire, dict grows unbounded.

### Bug 5 — Integer parsing without error handling (lines 858, 887, 952) 🟡 MEDIUM
`int(request.args.get("limit", 20))` → ValueError on bad input → 500 error.

---

## Model Test Results

**Benchmark task**: Fix Bug 2 (SRT→VTT) — self-contained, clearly defined.  
**Method**: `PlanActVerifyAgent.implement_proposals()` with `max_heal_iterations=2`, `num_ctx=30720`

| Model | Result | Time | Steps | Heals | Naive Removed | Notes |
|-------|--------|------|-------|-------|---------------|-------|
| deepseek-coder:33b-instruct-q4_K_M | ❌ FAILED | 201s | 8 | 2 | NO | JSON parse errors on plan output; edit steps failed (text not found); heal plan also produced malformed JSON |
| qwen2.5-coder:32b | ⏳ PENDING | — | — | — | — | — |
| qwen3:32b | ⏳ PENDING | — | — | — | — | — |

### Detailed Results

#### deepseek-coder:33b-instruct-q4_K_M (19GB)
- **Planning**: 60s to generate plan (2 steps: read + edit)
- **Edit attempt**: Failed immediately — "Text to replace not found"
- **Heal 1**: Generated 6-step plan but all 3 edit steps failed (text not found)
- **Heal 2**: JSON parse error in heal plan itself
- **Verdict**: Poor structured output compliance. Generates plans but can't produce valid edit blocks that match actual file content. JSON formatting issues in heal responses.

---

## Test Script Location
`/tmp/qa_codator_test.py` on pluton

## How to Resume Testing
```bash
# SSH to pluton
ssh marcin@192.168.50.119

# Run next model test
cd ~/git/coder && source .venv/bin/activate
python /tmp/qa_codator_test.py "qwen2.5-coder:32b"
python /tmp/qa_codator_test.py "qwen3:32b"
```

## Status: IN PROGRESS — 1 of 3 models tested
