# Data Format

RATIO expects fixed JSONL splits exported by the companion PACT repository.
Keep full datasets outside git; the local `data/` directory is ignored.

## Stage 1

Stage 1 trains the proportion-aware detector on PACT mixed-authorship records.
The maintained scripts expect:

```text
data/pact/train.jsonl
data/pact/val.jsonl
data/pact/test.jsonl
```

Required fields:

```json
{
  "id": "sample_id",
  "original_text": "human source document",
  "mixed_text": "document used as detector input",
  "target_ai_ratio": 0.4
}
```

Recommended PACT fields:

```json
{
  "sentence_labels": [0, 1, 1, 0, 0],
  "lir": 0.37,
  "jaccard_distance": 0.24,
  "sentence_jaccard": 0.31,
  "cosine_distance": 0.19,
  "source_dataset": "openwebtext",
  "source_domain": "web",
  "mixing_mode": "block_replace",
  "rewrite_model": "qwen3.5-flash"
}
```

`target_ai_ratio` is mapped to six bins: `0.0`, `0.2`, `0.4`, `0.6`, `0.8`,
and `1.0`. When available, continuous labels are used for the LIR target and
optional metadata/auxiliary losses.

## Stage 2

Stage 2 trains RCDPO on clean/rewrite pairs. The maintained scripts expect:

```text
data/rewrite/train.jsonl
data/rewrite/val.jsonl
data/rewrite/test.jsonl
```

Required fields:

```json
{
  "id": "sample_id",
  "original_text": "human source document",
  "mixed_text": "original mixed-authorship document",
  "rewritten_text": "rewritten or humanized document",
  "target_ai_ratio": 0.4
}
```

Recommended PACT rewrite fields:

```json
{
  "sentence_labels": [0, 1, 1, 0, 0],
  "rewrite_info": {"status": "ok", "rewriter": "qwen3.5-flash"},
  "mixed_lir": 0.37,
  "mixed_jaccard_distance": 0.24,
  "mixed_sentence_jaccard": 0.31,
  "mixed_cosine_distance": 0.19,
  "rewrite_lir": 0.28,
  "rewrite_jaccard_distance": 0.18,
  "rewrite_sentence_jaccard": 0.25,
  "rewrite_cosine_distance": 0.14
}
```

The Stage-2 loader filters records that do not contain both `mixed_text` and
`rewritten_text`. With attacked-only evaluation, records are considered attacked
when `target_ai_ratio > 0`, `rewrite_info.status == "ok"` when present, and the
rewrite is non-empty and different from `mixed_text`.

## PACT Connection

In the local project layout used for development:

```text
../PACT/datasplit/benchmark_grouped/{train,val,test}.jsonl
../PACT/datasplit/rewrite_grouped/{train,val,test}.jsonl
```

map to:

```text
data/pact/{train,val,test}.jsonl
data/rewrite/{train,val,test}.jsonl
```

Use a symlink or copy. Do not commit the dataset files into RATIO.
