<p align="center"><img src="assets/logo.png" width="96" alt="Ragentile logo"></p>

# Jasper-Harness

**Inference harnesses for the Jasper-K models for GPU kernel optimization.**

**[Technical blog](https://ragentile.github.io/Jasper-K/)** · **[Hugging Face](https://huggingface.co/ragentile)** · [Models and harnesses](#models-and-harnesses) · [Getting started](#getting-started)

Jasper-K is a family of models developed by **Ragentile** to generate correct GPU kernels and improve their performance through execution feedback. The family includes Jasper-K-27B, Jasper-K-32B, and Jasper-K-35B-MoE, obtained by RL post-training three different base models.

Model checkpoints are available from the linked Hugging Face repositories.

## Models and harnesses

| Model | Base model | Weights | Model card and harness |
|---|---|---|---|
| Jasper-K-27B | Qwen3.8-27B | [Hugging Face](https://huggingface.co/ragentile/Jasper-K-27B) | [27B harness](harnesses/Jasper-K-27B/) |
| Jasper-K-32B | QwQ-32B | [Hugging Face](https://huggingface.co/ragentile/Jasper-K-32B) | [32B harness](harnesses/Jasper-K-32B/) |
| Jasper-K-35B-MoE | Qwen3.6-35B-A3B | [Hugging Face](https://huggingface.co/ragentile/Jasper-K-35B-MoE) | [35B-MoE harness](harnesses/Jasper-K-35B-MoE/) |

Jasper-K-27B is trained from Qwen3.8-27B with the base model's reasoning effort set to medium. Each harness retains its model-specific generation budgets, sampling settings, and multi-turn behavior.

## RL post-training

We post-train Jasper-K using **multi-rollout, multi-turn GRPO**. The model generates candidate kernels, receives compilation, correctness, and measured-performance feedback, and revises its implementations over successive turns. Different trajectories explore different optimization strategies, while improvement-based rewards encourage continued search after finding a correct implementation.

Training uses **180 KernelBenchFixed Level 1 and Level 2 tasks**, with another 20 reserved for training monitoring and checkpoint selection. KernelBenchFixed revises tasks that can otherwise admit shortcuts.

**LeetCuda is our in-house benchmark of 50 kernel optimization tasks**, originating from the reinforcement learning team at the University of Alberta, Canada. Its tasks are used only for testing and are excluded from RL training and checkpoint selection.

## Results reported in the technical blog

| Model | KernelBench: base → Jasper-K | Relative gain | LeetCuda: base → Jasper-K | Relative gain |
|---|---:|---:|---:|---:|
| Jasper-K-27B | 2.289× → 3.471× | +51.64% | 2.161× → 2.580× | +19.39% |
| Jasper-K-32B | 1.265× → 1.301× | +2.85% | 1.323× → 1.399× | +5.74% |
| Jasper-K-35B-MoE | 1.308× → 1.345× | +2.83% | 1.373× → 1.610× | +17.26% |

Scores are **METR Best GM Speedup @16 × 4**. For each task, select the fastest correct implementation found across 16 trajectories of up to four turns; use the reference implementation when no faster correct candidate is found; then take the geometric mean across tasks. Comparisons use the same hardware platform, execution environment, correctness checks, timing protocol, and search budget.

KernelBench covers 250 tasks and includes training tasks. LeetCuda is held out. Under the reported LeetCuda settings, Jasper-K-27B performs competitively with the evaluated frontier models on kernel optimization. These are the blog's reported results under its evaluation settings, not a benchmark rerun of the release configurations here.

See the [technical blog](https://ragentile.github.io/Jasper-K/) for the full evaluation protocol, training method, and real-workload experiments.

## Getting started

Use Python 3.11 or newer, with vLLM and Transformers versions compatible with the chosen model architecture and your accelerator environment. The harness uses a vLLM `/v1/completions` endpoint with actual prompt and completion token IDs.

Clone the repository and choose a model. For example:

```bash
git clone https://github.com/Ragentile/Jasper-Harness.git
cd Jasper-Harness/harnesses/Jasper-K-27B

python3 -m pip install --upgrade huggingface_hub
hf download ragentile/Jasper-K-27B --local-dir ./model
```

The default checkpoint path in each GitHub harness is `./model`. You can instead set `[model.serve].path` in `harness.toml` to an existing checkpoint directory. Adjust device assignments, tensor parallelism, replicas, and ports for your hardware. See the [Hugging Face download instructions](https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-files) for CLI options.

### Connect your evaluation backend

[Evaluator template and runnable feedback examples](harnesses/Jasper-K-27B/examples/README.md) show the HTTP contract and the exact feedback text sent to the model.

**The evaluation backend is not included.** Provide an evaluator that compiles and runs candidates, checks numerical correctness against the reference, and measures execution time. The included adapter only connects this evaluator to the harness; it does not implement correctness checks, timing, profiling, or execution isolation.

Follow [EVALUATOR_API.md](EVALUATOR_API.md). Inside the chosen harness directory, copy `example_evaluator.py` to `my_evaluator.py` and implement `run_backend(request)`, then start the adapter:

```bash
python3 evaluator_adapter.py --evaluator my_evaluator:evaluate --host 127.0.0.1 --port 9000
```

Example invocation from the same harness directory; adapt the device IDs and evaluator URL to your environment:

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

Task files follow the KernelBench structure, with a reference `Model` and input-construction functions. Each trajectory retains its fastest verified-correct implementation. Without an evaluator endpoint, each trajectory generates only one turn.

Profiling is enabled by default. If your backend does not provide profiler summaries, use `--eval-profiling false` to exercise the loop; this changes the feedback conditions and is not a reproduction of the reported benchmark setup.

## Self-test

Each harness includes `selftest.sh`, `selftest_judge.py`, and small sample problems. Configure the model path and serving settings, start your own evaluator, and run from the chosen harness directory:

```bash
# Example device IDs; adjust to your model-serving configuration.
export JASPER_K_EVALUATOR_ENDPOINT="http://127.0.0.1:9000"
./selftest.sh --gpus 0,1,2,3,4,5,6,7 --samples 1 --timeout 7200
```

The self-test generates candidate kernels and re-evaluates them using your backend. It needs model serving and an implemented evaluator. It recreates `selftest_out` in the harness directory, and its results are not official benchmark scores. For a CPU-only protocol demonstration, use the [evaluator and feedback examples](harnesses/Jasper-K-27B/examples/README.md).

GitHub hosts the harness, templates, examples, and self-test. Hugging Face hosts model weights and their configuration/tokenizer files.

## License

The Jasper-K harness code and model weights are released under **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Base models and external dependencies retain their respective licenses and copyright notices.
