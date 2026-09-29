# Aurora AutoML

A local, bounded neural-architecture-search tool for synthetic tabular classification. It evolves small PyTorch multilayer perceptrons and ranks them by held-out accuracy relative to inference latency and parameter memory.

## Guarantees and boundaries

- **Local only:** the default workload creates synthetic tensors in memory. The program has no downloader, web client, plugin loader, shell execution, or source-file scanner.
- **Bounded search:** depth, widths, mutation rates, population size, and training work are constrained by CLI validation and genome validation.
- **Hardware aware:** `--device auto` chooses CUDA, then MPS, then CPU. Multiple local CUDA GPUs use `DataParallel`; no networked distributed runtime is started.
- **Safe artifacts:** results are atomically written as human-readable JSON. Checkpoints contain only model tensors and metadata; `load_champion` uses PyTorch's `weights_only=True` loading mode and schema validation.

## Install

Install a PyTorch wheel appropriate for the machine, then install the declared dependency:

```bash
python -m pip install -r requirements.txt
```

See [PyTorch's installation selector](https://pytorch.org/get-started/) when CUDA or Apple Silicon support is required.

## Run a bounded search

```bash
python aurora_automl.py \
  --generations 4 --population 8 --elites 3 --epochs 5 \
  --samples 2400 --features 12 --device auto \
  --output artifacts/aurora_results.json \
  --checkpoint artifacts/aurora_champion.pt
```

The Aurora terminal dashboard reports the best genome, accuracy, per-sample latency, estimated parameter memory, fitness, and fitness convergence for every generation. Use `python aurora_automl.py --help` for all workload controls.

## Reuse a champion locally

```python
import torch
from aurora_automl import load_champion

model, genome = load_champion("artifacts/aurora_champion.pt", "cpu")
with torch.inference_mode():
    logits = model(torch.randn(2, 12))
```

The input feature count must match the search that produced the checkpoint. `genome.describe()` prints the retained architecture and optimizer DNA.
