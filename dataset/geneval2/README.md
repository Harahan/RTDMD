# GenEval2 dataset (prepared copy)

- `train.jsonl`: 20000 synthetic training prompts (disjoint from official benchmark).
- `test.jsonl`: 800 official GenEval2 benchmark prompts (for evaluation).
- `merged.jsonl`: concatenation of train + test (20800), used for prompt-to-vqa lookup.

These files are copied from:
`/yushihuang2/Mask/Flow-Factory/dataset/GenEval2/synthetic/`.

Use in FlowDMD:
- training prompts: `prompt_path: "dataset/geneval2"` (auto-detects `train.jsonl`)
- eval prompts: `eval.dataset: "geneval2"` with `eval.eval_prompt_source: "dataset"`
- reward: `reward_fn: {"geneval2": 1.0}`

GenEval2 reward model resolution priority:
1. explicit `FLOWDMD_GENEVAL2_MODEL_NAME`
2. `reward_ckpt_path` (if set in YAML) as:
   - direct model dir, or
   - `<reward_ckpt_path>/geneval2`, or
   - `<reward_ckpt_path>/geneval2/Qwen3-VL-8B-Instruct`, or
   - `<reward_ckpt_path>/Qwen3-VL-8B-Instruct`
3. fallback to HF Hub id: `Qwen/Qwen3-VL-8B-Instruct`
