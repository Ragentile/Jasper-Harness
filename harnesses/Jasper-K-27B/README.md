# Jasper-K-27B

Jasper-K-27B is a 27B-parameter model obtained by RL post-training Qwen3.8-27B with the base model's reasoning effort set to medium, developed by **Ragentile** for GPU kernel optimization. It learns to generate correct kernel implementations and improve their performance through repeated interaction with an execution environment.

**[Read the Jasper-K technical blog](https://ragentile.github.io/Jasper-K/)** for the training method, evaluation results, and real-workload experiments.

## RL post-training

We post-train Jasper-K using **multi-rollout, multi-turn GRPO**: the model generates candidate kernels, receives compilation, correctness, and measured-performance feedback, and revises its implementations over successive turns. Across trajectories, it explores different optimization strategies; improvement-based rewards encourage it to keep searching for faster implementations after finding a correct one.

RL training uses **180 KernelBenchFixed Level 1 and Level 2 tasks**, with another 20 tasks reserved for training monitoring and checkpoint selection. KernelBenchFixed revises tasks that can otherwise admit shortcuts. **LeetCuda is our in-house benchmark of 50 kernel optimization tasks**, originating from the reinforcement learning team at the University of Alberta, Canada. Its tasks are used only for testing and are excluded from RL training and checkpoint selection.

## Results reported in the technical blog

| Benchmark | Qwen3.8-27B | Jasper-K-27B | Relative improvement |
|---|---:|---:|---:|
| KernelBench (250 tasks, L1–L3) | 2.289× | 3.471× | +51.64% |
| LeetCuda (50 tasks) | 2.161× | 2.580× | +19.39% |

Scores are **METR Best GM Speedup @16 × 4**: for each task, select the fastest correct implementation found across 16 trajectories of up to four turns, use the reference implementation when no faster correct candidate is found, and take the geometric mean across tasks. The reported comparisons use the same hardware platform, execution environment, correctness checks, timing protocol, and search budget. KernelBench includes training tasks; LeetCuda is held out.

Under the reported LeetCuda settings, Jasper-K-27B performs competitively with the evaluated frontier models on kernel optimization tasks. These are the blog's reported results under its evaluation settings, not a benchmark rerun of the release configuration below.

Other models in the family: [Jasper-K-32B](https://huggingface.co/ragentile/Jasper-K-32B) · [Jasper-K-35B-MoE](https://huggingface.co/ragentile/Jasper-K-35B-MoE).

## Jasper-K inference harness

**Included: multi-turn, multi-rollout inference code. Download the model weights from [Hugging Face](https://huggingface.co/ragentile/Jasper-K-27B). The evaluation backend is not included.**

Supply your own correctness checks, performance measurements, and execution isolation, and connect them through the generic adapter described in [EVALUATOR_API.md](EVALUATOR_API.md). The adapter itself does not implement compilation, device execution, timing, or profiling.

## Configuration

| Setting | Value |
|---|---|
| Model identifier | `Jasper-K-27B` |
| Prompt and feedback format | `Jasper-K` |
| Total generation budget | 61,440 tokens per turn |
| First-stage thinking budget | 49,152 tokens |
| Temperature / top_p | 1.0 / 0.95 |
| Server context window | 122,880 tokens |
| Maximum turns per trajectory | 4 |
| Profiling | Enabled |

This harness uses the Jasper-K prompts, rules, numerical-precision instructions, and multi-turn feedback. A truncated generation receives dedicated truncation feedback before the next turn.

When `top_k=-1`, this implementation omits `top_k` from the request, so the server or model default applies. The original checkpoint defaults to `top_k=20`. The request explicitly sets `reasoning_effort=medium`.

The thinking budget limits the length of the first generation stage. If the model finishes naturally, its result is used directly. If the stage reaches its limit without `</think>`, the harness inserts a closing instruction and continues the answer. This is not an exact count of tokens containing only reasoning.

## Environment and model directory

- Python 3.11 or newer.
- vLLM and Transformers versions compatible with the model architecture and your accelerator environment.
- A vLLM `/v1/completions` endpoint that supports `return_token_ids` and returns actual prompt and completion token IDs.
- Your own evaluator, exposed through the adapter's `/evaluate` endpoint.

Set `[model.serve].path` in `harness.toml` to the complete checkpoint directory. The default path is `./model`. Download the checkpoint, configuration, and tokenizer files into that directory using `hf download ragentile/Jasper-K-27B --local-dir ./model`. It must include weight shards, `config.json`, the tokenizer, and a chat template. Relative paths are resolved against the harness directory.

Adjust tensor parallelism (`tp`), the number of replicas, and ports for your hardware. When running multiple models, assign separate devices and service ports.

## Running the harness

See [Evaluator template and runnable feedback examples](examples/README.md) for an implementation template, complete response fixtures, and the exact feedback text sent to the model.

Example using your own evaluator: implement the evaluation function following `EVALUATOR_API.md`, start its adapter, and configure the checkpoint path and serving settings in `harness.toml`. Adapt the device IDs, evaluator URL, and task directory below to your environment:

```bash
# Example device IDs; match tp and replicas in harness.toml to your hardware.
export JASPER_K_GPUS="0,1,2,3,4,5,6,7"
# Example URL for your own implemented evaluator adapter.
export JASPER_K_EVALUATOR_ENDPOINT="http://127.0.0.1:9000"

python3 harness.py \
  --problems /absolute/path/to/problems \
  --out ./run_out \
  --backend cuda --samples 16 --timeout 7200
```

Tasks are organized as `level1/1_example.py`, `level2/2_example.py`, and so on. Each task file contains a reference `Model` and input-construction functions.

`--samples` is the number of independent trajectories per task. The number of turns follows the configuration and each model's stopping rules. **Without an evaluator endpoint, each trajectory generates only one turn; execution-feedback-driven multi-turn optimization is unavailable.**

`--samples 16` requests 16 independent rollouts per task. With `loop.max_rounds=4`, each rollout can take up to four turns, giving at most 64 generation/evaluation rounds per task. Truncation, context limits, and timeouts may stop a trajectory earlier. Its history, code, and evaluation results are passed to the next turn, and it independently retains its fastest verified-correct implementation.

The default run produces 16 candidates per task. Aggregate best-of-N results across trajectories using the same evaluator. `selftest_judge.py` can re-evaluate saved candidates and aggregate results within each task.

`JASPER_K_EVALUATOR_ENDPOINT` specifies the evaluator adapter used for execution feedback. Each trajectory writes a separate source file, with a summary in `run_out/_harness_results.json`.

Example self-test:

```bash
./selftest.sh --gpus 0,1,2,3,4,5,6,7 --samples 1 --timeout 7200
```

The self-test recreates `selftest_out` in this directory. Its final re-evaluation disables profiling. The reported scores depend on your evaluator settings and serve as a local integration check.

## Validation and interpretation

Use `selftest.sh` and `selftest_judge.py` with your model service and evaluator for a local integration check. The included sample problems exercise the generation and evaluation flow. The [feedback examples](examples/README.md) also provide a CPU-only HTTP protocol demonstration with clearly labeled synthetic results.

## Implementation

`kernel_protocol.py` provides prompt templates, response parsing, and feedback formatting. `solve.py` implements the optimization loop, and `harness.py` runs trajectories across tasks.

## License

The Jasper-K-27B model weights and harness code are released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution. The base model is `Qwen/Qwen3.8-27B`; external dependencies and base models retain their respective licenses and copyright notices.
