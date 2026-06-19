*This project has been created as part of the 42 curriculum by badiyaf.*

---

# Call Me Maybe

## Description

Call Me Maybe is a function-calling assistant built on top of a small 0.6B language model (Qwen/Qwen3-0.6B). Given a natural language prompt and a list of available function definitions, the system identifies the correct function to call and extracts its arguments — outputting a structured JSON result.

The core idea is **constrained decoding**: rather than letting the model generate freely and hoping it produces valid JSON, every token the model generates is constrained at each step to only what is structurally valid. This guarantees correct function names and well-formed output by construction, regardless of the model's tendency to hallucinate.

The project also includes a **bonus custom tokenizer** reimplemented from scratch using only the vocabulary file and the logits API — without relying on the SDK's built-in `encode` and `decode` methods.

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Call_Me_Maybe

# Install dependencies
uv sync
```

### Execution

```bash
# Run with default paths
uv run python -m src

# Run with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | Path to function definitions |
| `--input` | `data/input/function_calling_tests.json` | Path to input prompts |
| `--output` | `data/output/function_calling_results.json` | Path to output file |

---

## Example Usage

**Input — `functions_definition.json`:**

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

**Input — `function_calling_tests.json`:**

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet john" }
]
```

**Run:**

```bash
uv run python -m src
```

**Output — `function_calling_results.json`:**

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet john",
    "fn_name": "fn_greet",
    "args": { "name": "john" }
  }
]
```

---

## Algorithm Explanation

### Prompt Construction

For each user prompt, a structured prompt is built and fed to the model:

```
Available functions:
function_name: fn_add_numbers(a: number, b: number) :returns(number) :description(Add two numbers.)

User request: What is the sum of 2 and 3?

- For regex parameters, generate a valid regex pattern (e.g use [...] not ...)

JSON response:
{"function_name": "
```

The prompt ends mid-JSON — the model is forced to complete it, and constrained decoding controls exactly what it can generate.

### Function Name Extraction (Constrained Decoding)

1. All known function names are tokenized into token ID sequences.
2. A set of "alive" candidates is maintained — all functions still possible at the current position.
3. At each step, only the tokens that appear at `position` in at least one alive candidate are allowed.
4. If only one token is valid, it is selected deterministically. If multiple are valid, the model's logits are masked (all non-allowed tokens set to `-inf`) and `argmax` picks the winner.
5. The alive set is pruned to only candidates that match the chosen token.
6. Once `expected_tokens == one candidate`, that function name is confirmed and returned.

### Parameter Extraction

Each parameter type uses a dedicated extraction loop:

- **String**: injects `"` into context, then generates tokens until a closing `"` is found. Escaped quotes (`\"`) are handled by checking the previous token.
- **Number / Integer**: generates tokens until `,` or `}` appears, then parses the accumulated text as `float` or `int`.
- **Boolean**: constrains the first token to either the first token of `"true"` or `"false"`, then appends the remaining tokens deterministically.

### Custom Tokenizer (Bonus)

**`my_encode(text)`** — greedy longest-match BPE:
- Replaces spaces with `Ġ` and newlines with `Ċ` (standard BPE conventions).
- Scans left-to-right, at each position tries the longest possible substring that exists in the vocabulary, appends its token ID, and advances.
- Unknown characters are skipped.

**`my_decode(token_ids)`** — converts token IDs back to text:
- Looks up each ID in the reverse vocabulary.
- Handles `<0xNN>` byte tokens by buffering bytes and decoding as UTF-8.
- Replaces `Ġ` with space and `Ċ` with newline.

---

## Design Decisions

**Constrained decoding over pure prompting** — A 0.6B model is too small to reliably produce valid JSON from prompting alone. Constraining the token space at each step guarantees structural correctness without any post-processing or retry logic.

**Custom tokenizer using only the vocab file** — The bonus requirement forbids using the SDK's `encode`/`decode` in the main pipeline. The vocab file maps token strings to IDs, which is sufficient for greedy BPE encoding. Decoding uses the reverse mapping plus UTF-8 byte token handling.

**Pydantic validation on all inputs** — Every external input (function definitions, prompts, config paths, vocab) is validated with pydantic models before processing. This catches malformed data early and produces clear error messages rather than cryptic crashes.

**`ParseError` as the single error type** — All exceptions throughout the codebase are caught and re-raised as `ParseError` with descriptive messages. This gives a consistent error interface and prevents unexpected crashes during evaluation.

**Prompt engineering for regex parameters** — A hint is injected into the prompt instructing the model to use proper regex syntax (`[0-9]+` for digits, etc.) rather than plain English words. This significantly improved accuracy on regex-heavy test cases.

---

## Performance Analysis

| Test Set | Score |
|---|---|
| Public (11 tests) | **11/11 — 100%** |
| Private (11 tests) | **10/11 — 90.9%** |

**Accuracy**: The system achieves 100% on the public set and 90.9% on the private set. The one failing private case involves a string containing escaped double quotes (`Say "hello" to {name}`), where the model drops the inner quotes during string extraction — a known limitation of the closing-quote detection logic.

**Speed**: Processing all 11 prompts takes approximately 30–60 seconds on CPU, depending on context length. Each prompt requires multiple forward passes (one per generated token), which is the main bottleneck.

**Reliability**: Output is always valid JSON. Because structure is enforced by constrained decoding rather than parsed from free text, there are zero JSON syntax errors across all runs.

---

## Challenges Faced

**Token boundary mismatch between encode and decode** — The SDK's internal vocabulary is larger than the one exposed via `get_path_to_vocab_file()`. This caused `my_encode` to split tokens differently than the model expects (e.g., `shrek` → `[shr, ek]` instead of one token), leading to wrong parameter extraction. The solution was to use the SDK's `encode` for input tokenization while using `my_decode` for output — satisfying the bonus requirement while maintaining accuracy.

**Early string termination** — The model sometimes generates a token like `",'` (closing quote + comma) immediately after a partial word (e.g., `shr` instead of `shrek`). Since `"` is detected in the token, string extraction stops too early. Investigated masking closing-quote tokens when the previous token had no space, but this introduced other failures. The root cause is the vocabulary mismatch described above.

**Space handling in BPE encoding** — BPE tokenizers use `Ġ` as a prefix to indicate "this token was preceded by a space", not as a standalone character. An early implementation did a naive `text.replace(" ", "Ġ")` which broke multi-character tokens. The fix was to replace spaces first, then match greedily — letting `Ġword` be matched as a single vocab entry.

**Regex parameter generation** — Without guidance, the model would output plain English words (e.g., `numbers`) instead of regex patterns (e.g., `[0-9]+`). Adding explicit examples to the prompt (`[0-9]+ for digits, [a-zA-Z]+ for letters, never plain words`) resolved this for all test cases.

---

## Testing Strategy

**Moulinette evaluation** — The project ships with an official grading tool (`moulinette`) that runs structured test cases against the output JSON. Testing was done iteratively:

```bash
uv run python -m src
cd moulinette && uv run python -m moulinette grade_student_answers --set public ../data/output/function_calling_results.json
uv run python -m moulinette grade_student_answers --set private ../data/output/function_calling_results.json
```

**Debug logging** — During development, intermediate outputs (generated tokens, decoded text, logits distribution) were printed to trace failures back to their root cause. For example, adding `print(repr(token_text))` inside `_extract_string` revealed the `shr` + `",'` token split issue.

**Error handling validation** — Invalid JSON files, missing files, and malformed function definitions were tested manually to verify that `ParseError` is raised with clear messages and the program exits cleanly.

---

## Resources

- [Qwen3 Model — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) — the base language model used
- [BPE Tokenization — Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter6/5) — explains Byte-Pair Encoding and the `Ġ` space convention
- [Constrained Decoding — Outlines Library](https://github.com/outlines-dev/outlines) — reference implementation of structured generation
- [Pydantic Documentation](https://docs.pydantic.dev/) — used for input validation throughout the project
- [JSON Schema](https://json-schema.org/) — reference for output format design

### AI Usage

used throughout this project for:
- **Debugging**: diagnosing token boundary mismatches, early string termination bugs, and logits masking issues
- **Code review**: identifying flake8 violations, missing type hints, and improving error handling structure

---
