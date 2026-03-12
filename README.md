# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Fork note: this repo is an experiment

This fork keeps the original "let an agent iterate on `train.py` under a fixed 5-minute budget" idea from [karpathy/autoresearch](https://github.com/karpathy/autoresearch/tree/master), but repurposes it for a different task: binary sequence classification on preprocessed text tensors instead of language-model pretraining on climbmix.

In practice, this means the current repo is best read as an `autoresearch`-style research harness for a small NLP classification problem. Some upstream files and wording are still present for reference, but the active experiment path in this fork is the classifier in `train.py`.

## What changed from the original repo

Compared with the upstream repository:

- **Objective:** upstream optimizes `val_bpb` for next-token prediction; this fork optimizes `val_accuracy` for binary classification.
- **Data path:** upstream expects `prepare.py` to download climbmix shards and train a tokenizer; this fork trains from checked-in `train_data.pt` and `val_data.pt`, with source CSVs in `technology.csv` and `data_for_preprocessing.csv`.
- **Model behavior:** upstream uses a causal GPT for language modeling; this fork switches to non-causal, bidirectional attention plus padding-aware pooling for sequence classification.
- **Sequence budget:** upstream runs at `MAX_SEQ_LEN=2048`; this fork reduces that to `256`, which makes faster, smaller classification experiments practical.
- **Outputs:** upstream logs `val_bpb`, MFU, token throughput, and parameter count; this fork reports `val_accuracy`, runtime, VRAM, step count, and depth.
- **Artifacts:** this fork adds experiment outputs such as `results.png` and keeps local tensor datasets in-repo to shorten setup.

## Current experiment setup

The current `train.py` is a single-file autonomous tuning target for a Reddit/news-style text classification workflow:

- input batches come from `train_data.pt` and `val_data.pt`
- labels are binary
- training still runs with a fixed 5-minute wall-clock budget
- the model is a GPT-style transformer adapted for classification
- the main score to beat is `val_accuracy`

So the repo still matches the original autoresearch spirit, but the experiment itself is no longer "train a better tiny language model overnight"; it is "let the agent search for a better classifier under a fixed time budget".

## Results so far

This fork already includes one short autonomous tuning run, summarized in `results.png` and reflected by the recent experiment commits on the `autoresearch/mar12-gpu` branch that was later merged back to `master`.

- 5 experiments were recorded
- 1 experiment was kept and 4 were discarded
- 0 crashes were recorded
- best observed validation accuracy was **0.8125** on commit `c102a9a`
- peak memory ranged from roughly **0.2 GB** to **1.6 GB** across the tested variants

From the commit history, the explored changes included:

- adding PAD-aware attention pooling (`attn_last`)
- lowering weight decay
- increasing depth to 4
- scaling to depth 6 / width 384 with lower learning rates

At least in this first pass, the baseline configuration remained the best result, which is a useful outcome in itself: the autoresearch loop quickly rejected several plausible modifications without beating the starting point.

Note: `results.tsv` is referenced by the analysis notebook and agent workflow, but it is not currently present in this checkout. The summary above is based on the persisted experiment artifacts already in the repo (`results.png` plus recent experiment commits).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — mostly retained from upstream; contains tokenizer/data utilities for the original pretraining setup and is not the main path for this fork's checked-in classification tensors.
- **`train.py`** — the single file the agent edits. In this fork it contains the GPT-style classifier, optimizer (Muon + AdamW), and 5-minute training loop. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. In the original repo the metric is **val_bpb**; in this fork the active score is **val_accuracy** on the held-out classification set.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run a single classification experiment (~5 min)
uv run train.py
```

In this fork, the preprocessed tensors are already checked into the repo, so `prepare.py` is not part of the default training path.

If you want to compare against the original upstream workflow, that original setup is still documented in [karpathy/autoresearch](https://github.com/karpathy/autoresearch/tree/master), where `prepare.py` downloads climbmix data and trains the tokenizer used for language-model experiments.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py                 — upstream-style data/tokenizer utilities kept for reference
train.py                   — classification model, optimizer, and 5-minute training loop
program.md                 — agent instructions for autonomous experiments
technology.csv             — source dataset slice
data_for_preprocessing.csv — preprocessing input data
train_data.pt              — preprocessed training tensors
val_data.pt                — preprocessed validation tensors
results.png                — experiment plot
pyproject.toml             — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
