#!/usr/bin/env python3
"""Aurora AutoML: safe local neuroevolution for tabular classification.

The program uses only generated local data by default. It never executes files,
downloads code, or inspects locations outside the paths supplied on its CLI.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# ANSI Aurora palette: Polar Night, Arctic Sky, Aurora Green, Twilight Purple, Orchid.
class Aurora:
    RESET = "\033[0m"; BOLD = "\033[1m"; POLAR = "\033[38;5;254m"
    SKY = "\033[38;5;117m"; GREEN = "\033[38;5;84m"; PURPLE = "\033[38;5;141m"; ORCHID = "\033[38;5;213m"
    @classmethod
    def paint(cls, value: object, color: str, bold: bool = False) -> str:
        return f"{cls.BOLD if bold else ''}{color}{value}{cls.RESET}"
    @classmethod
    def info(cls, message: str) -> None: print(cls.paint("◆ ", cls.SKY, True) + message)
    @classmethod
    def success(cls, message: str) -> None: print(cls.paint("✓ ", cls.GREEN, True) + message)
    @classmethod
    def mutation(cls, message: str) -> None: print(cls.paint("✦ ", cls.PURPLE, True) + message)
    @classmethod
    def alert(cls, message: str) -> None: print(cls.paint("! ", cls.ORCHID, True) + message)

ACTIVATIONS = ("relu", "gelu", "silu", "tanh")
NORMALIZATIONS = ("none", "batch", "layer")

def activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}[name]

@dataclass
class Genome:
    """Serializable architecture and optimizer DNA with bounded, valid genes."""
    widths: list[int]
    activations: list[str]
    dropout: list[float]
    normalizations: list[str]
    learning_rate: float
    weight_decay: float
    identifier: str = field(default_factory=lambda: f"g-{random.getrandbits(32):08x}")

    @classmethod
    def random(cls, rng: random.Random, min_depth: int, max_depth: int) -> "Genome":
        depth = rng.randint(min_depth, max_depth)
        widths = [rng.choice((16, 24, 32, 48, 64, 96, 128, 192, 256)) for _ in range(depth)]
        return cls(widths, [rng.choice(ACTIVATIONS) for _ in widths],
                   [round(rng.uniform(0, .40), 2) for _ in widths],
                   [rng.choice(NORMALIZATIONS) for _ in widths],
                   10 ** rng.uniform(-3.8, -1.5), 10 ** rng.uniform(-6, -2.5))

    def validate(self, min_depth: int = 1, max_depth: int = 8) -> None:
        n = len(self.widths)
        if not min_depth <= n <= max_depth: raise ValueError("Genome depth is outside safe bounds")
        if not (n == len(self.activations) == len(self.dropout) == len(self.normalizations)): raise ValueError("Layer gene lengths differ")
        if any(w < 4 or w > 1024 for w in self.widths): raise ValueError("Invalid layer width")
        if any(a not in ACTIVATIONS for a in self.activations): raise ValueError("Unknown activation")
        if any(not 0 <= d < .9 for d in self.dropout): raise ValueError("Invalid dropout")
        if any(nm not in NORMALIZATIONS for nm in self.normalizations): raise ValueError("Unknown normalization")
        if not 1e-6 <= self.learning_rate <= 1: raise ValueError("Invalid learning rate")
        if not 0 <= self.weight_decay <= 1: raise ValueError("Invalid weight decay")

    def describe(self) -> str:
        layers = " → ".join(str(x) for x in self.widths)
        return f"{self.identifier} [{layers}] lr={self.learning_rate:.2e}, wd={self.weight_decay:.1e}"

    def to_dict(self) -> dict: return dataclasses.asdict(self)
    @classmethod
    def from_dict(cls, value: dict) -> "Genome":
        genome = cls(**value); genome.validate(); return genome

class EvolvedNet(nn.Module):
    """Compiles valid DNA into a live PyTorch classifier."""
    def __init__(self, genome: Genome, input_features: int, classes: int):
        super().__init__(); genome.validate()
        blocks: list[nn.Module] = []; current = input_features
        for width, act, drop, norm in zip(genome.widths, genome.activations, genome.dropout, genome.normalizations):
            blocks.append(nn.Linear(current, width))
            if norm == "batch": blocks.append(nn.BatchNorm1d(width))
            elif norm == "layer": blocks.append(nn.LayerNorm(width))
            blocks.append(activation(act))
            if drop > 0: blocks.append(nn.Dropout(drop))
            current = width
        blocks.append(nn.Linear(current, classes)); self.network = nn.Sequential(*blocks)
    def forward(self, x: Tensor) -> Tensor: return self.network(x)

@dataclass
class Evaluation:
    genome: Genome
    accuracy: float
    latency_ms: float
    memory_mb: float
    parameters: int
    fitness: float
    state_dict: dict[str, Tensor] = field(repr=False, compare=False)

    def to_dict(self) -> dict:
        """Return JSON-safe result metadata without duplicating tensor checkpoints."""
        return {
            "genome": self.genome.to_dict(),
            "accuracy": self.accuracy,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "parameters": self.parameters,
            "fitness": self.fitness,
        }

class FitnessEvaluator:
    """Trains within explicit batch/epoch bounds and measures held-out inference."""
    def __init__(self, device: torch.device, epochs: int, batch_size: int, workers: int = 0):
        self.device, self.epochs, self.batch_size = device, epochs, batch_size
        self.workers = max(0, workers)
        self.amp = device.type == "cuda"

    def _loader(self, dataset: TensorDataset, shuffle: bool) -> DataLoader:
        # A final one-example training batch is incompatible with BatchNorm.  It
        # is safe to omit only that batch; validation always retains all rows.
        drop_last = shuffle and len(dataset) % self.batch_size == 1
        return DataLoader(dataset, self.batch_size, shuffle=shuffle, num_workers=self.workers,
                          pin_memory=self.device.type == "cuda", persistent_workers=self.workers > 0,
                          drop_last=drop_last)

    def evaluate(self, genome: Genome, train: TensorDataset, validation: TensorDataset, input_features: int, classes: int) -> Evaluation:
        model = EvolvedNet(genome, input_features, classes).to(self.device)
        # DataParallel uses only local GPUs and never contacts an untrusted endpoint.
        usable = torch.cuda.device_count() if self.device.type == "cuda" else 0
        parallel: nn.Module = nn.DataParallel(model) if usable > 1 else model
        optimizer = torch.optim.AdamW(parallel.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay)
        criterion = nn.CrossEntropyLoss(); scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        parallel.train()
        for _ in range(self.epochs):
            for x, y in self._loader(train, True):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp): loss = criterion(parallel(x), y)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        accuracy = self._accuracy(parallel, validation)
        latency = self._latency(parallel, validation.tensors[0][:min(128, len(validation))])
        params = sum(p.numel() for p in model.parameters())
        memory = params * 4 / 1024**2
        # Absolute performance/efficiency objective. The epsilon makes perfect tiny models finite.
        fitness = accuracy / max(.01, latency + memory)
        state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        return Evaluation(genome, accuracy, latency, memory, params, fitness, state_dict)

    @torch.inference_mode()
    def _accuracy(self, model: nn.Module, validation: TensorDataset) -> float:
        model.eval(); correct = count = 0
        for x, y in self._loader(validation, False):
            output = model(x.to(self.device, non_blocking=True)); correct += (output.argmax(1).cpu() == y).sum().item(); count += len(y)
        return correct / max(1, count)

    @torch.inference_mode()
    def _latency(self, model: nn.Module, sample: Tensor) -> float:
        model.eval(); x = sample.to(self.device)
        if self.device.type == "cuda":
            sync = torch.cuda.synchronize
        elif self.device.type == "mps":
            sync = torch.mps.synchronize
        else:
            sync = lambda: None
        for _ in range(3): model(x)
        sync(); starts = []
        for _ in range(10):
            sync(); start = time.perf_counter(); model(x); sync(); starts.append((time.perf_counter() - start) * 1000)
        return statistics.median(starts) / len(x)

class EvolutionEngine:
    def __init__(self, rng: random.Random, population_size: int, elite_count: int, mutation_rate: float, min_depth: int, max_depth: int):
        self.rng, self.population_size, self.elite_count = rng, population_size, elite_count
        self.mutation_rate, self.min_depth, self.max_depth = mutation_rate, min_depth, max_depth

    def initial_population(self) -> list[Genome]: return [Genome.random(self.rng, self.min_depth, self.max_depth) for _ in range(self.population_size)]
    def crossover(self, first: Genome, second: Genome) -> Genome:
        """Position-aware blend; layer genes remain aligned and therefore compilable."""
        target_depth = self.rng.randint(min(len(first.widths), len(second.widths)), max(len(first.widths), len(second.widths)))
        def inherited(seq_a: Sequence, seq_b: Sequence, i: int):
            return self.rng.choice((seq_a[min(i, len(seq_a)-1)], seq_b[min(i, len(seq_b)-1)]))
        return Genome([inherited(first.widths, second.widths, i) for i in range(target_depth)],
                      [inherited(first.activations, second.activations, i) for i in range(target_depth)],
                      [inherited(first.dropout, second.dropout, i) for i in range(target_depth)],
                      [inherited(first.normalizations, second.normalizations, i) for i in range(target_depth)],
                      math.sqrt(first.learning_rate * second.learning_rate), math.sqrt(first.weight_decay * second.weight_decay))
    def mutate(self, source: Genome) -> Genome:
        child = copy.deepcopy(source); child.identifier = f"g-{self.rng.getrandbits(32):08x}"
        def hit() -> bool: return self.rng.random() < self.mutation_rate
        if hit() and len(child.widths) < self.max_depth:
            i = self.rng.randrange(len(child.widths)+1); child.widths.insert(i, self.rng.choice((16,32,64,128,256))); child.activations.insert(i, self.rng.choice(ACTIVATIONS)); child.dropout.insert(i, round(self.rng.uniform(0,.4),2)); child.normalizations.insert(i, self.rng.choice(NORMALIZATIONS))
        if hit() and len(child.widths) > self.min_depth:
            i = self.rng.randrange(len(child.widths)); [seq.pop(i) for seq in (child.widths, child.activations, child.dropout, child.normalizations)]
        for i in range(len(child.widths)):
            if hit(): child.widths[i] = self.rng.choice((16,24,32,48,64,96,128,192,256))
            if hit(): child.activations[i] = self.rng.choice(ACTIVATIONS)
            if hit(): child.dropout[i] = round(min(.65, max(0, child.dropout[i] + self.rng.gauss(0,.10))), 2)
            if hit(): child.normalizations[i] = self.rng.choice(NORMALIZATIONS)
        if hit(): child.learning_rate = min(.1, max(1e-5, child.learning_rate * math.exp(self.rng.gauss(0,.5))))
        if hit(): child.weight_decay = min(.1, max(0, child.weight_decay * math.exp(self.rng.gauss(0,.7))))
        child.validate(self.min_depth, self.max_depth); return child
    def next_population(self, scored: list[Evaluation]) -> list[Genome]:
        ranked = sorted(scored, key=lambda e: e.fitness, reverse=True); elites = [copy.deepcopy(x.genome) for x in ranked[:self.elite_count]]
        children = elites[:]
        while len(children) < self.population_size:
            parents = self.rng.choices(elites, k=2); children.append(self.mutate(self.crossover(*parents)))
        return children

class Dashboard:
    def __init__(self, total: int): self.total, self.history = total, []
    def generation(self, index: int, values: list[Evaluation]) -> None:
        champion = max(values, key=lambda x: x.fitness); self.history.append(champion.fitness)
        trend = "".join("▁▂▃▄▅▆▇█"[min(7, int(v / max(self.history) * 7))] for v in self.history)
        border = Aurora.paint("═" * 76, Aurora.SKY)
        print(border); print(Aurora.paint(f"  AURORA NEUROEVOLUTION  ·  GENERATION {index}/{self.total}", Aurora.SKY, True))
        print(f"  Elite {Aurora.paint(champion.genome.identifier, Aurora.GREEN, True)}  accuracy {Aurora.paint(f'{champion.accuracy:.2%}', Aurora.GREEN)}  fitness {Aurora.paint(f'{champion.fitness:.5f}', Aurora.GREEN)}")
        print(f"  {champion.genome.describe()}\n  latency {champion.latency_ms:.4f} ms/sample · memory {champion.memory_mb:.3f} MB · params {champion.parameters:,}")
        print(Aurora.paint(f"  convergence  {trend}", Aurora.PURPLE)); print(border)

def synthetic_dataset(samples: int, features: int, seed: int) -> tuple[TensorDataset, TensorDataset, int]:
    generator = torch.Generator().manual_seed(seed); x = torch.randn(samples, features, generator=generator)
    # Nonlinear decision surface: meaningful NAS workload without network access.
    logits = x[:, 0] * x[:, 1] + .7 * torch.sin(x[:, 2]) - .4 * x[:, 3] ** 2 + .25 * x[:, 4:].sum(1)
    y = torch.bucketize(logits, torch.tensor([-0.7, .65]))
    order = torch.randperm(samples, generator=generator); cut = int(samples*.8)
    return TensorDataset(x[order[:cut]], y[order[:cut]]), TensorDataset(x[order[cut:]], y[order[cut:]]), 3

def choose_device(request: str) -> torch.device:
    if request != "auto":
        target = torch.device(request)
        if target.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
        if target.type == "mps" and not torch.backends.mps.is_available(): raise ValueError("MPS requested but unavailable")
        return target
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def save_result(path: Path, best: Evaluation, history: list[Evaluation], arguments: argparse.Namespace) -> None:
    """Persist only local, inspectable metadata using an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best": best.to_dict(),
        "history": [item.to_dict() for item in history],
        "config": {name: str(value) if isinstance(value, Path) else value for name, value in vars(arguments).items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, best: Evaluation, input_features: int, classes: int) -> None:
    """Save the champion weights and reconstruction metadata for local reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "aurora-automl-checkpoint-v1",
        "genome": best.genome.to_dict(),
        "input_features": input_features,
        "classes": classes,
        "state_dict": best.state_dict,
    }, temporary)
    temporary.replace(path)


def load_champion(path: Path, device: torch.device | str = "cpu") -> tuple[EvolvedNet, Genome]:
    """Safely rebuild a champion saved by :func:`save_checkpoint`.

    ``weights_only=True`` rejects arbitrary pickled objects, so this loader does
    not execute code embedded in a checkpoint.  It accepts only Aurora's own
    metadata schema and verifies the state dictionary against the rebuilt model.
    """
    payload: Any = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != "aurora-automl-checkpoint-v1":
        raise ValueError("Not an Aurora AutoML v1 checkpoint")
    genome_data = payload.get("genome")
    input_features, classes = payload.get("input_features"), payload.get("classes")
    state_dict = payload.get("state_dict")
    if (not isinstance(genome_data, dict) or not isinstance(input_features, int)
            or not isinstance(classes, int) or not isinstance(state_dict, Mapping)
            or input_features < 1 or classes < 2):
        raise ValueError("Aurora checkpoint metadata is malformed")
    genome = Genome.from_dict(genome_data)
    model = EvolvedNet(genome, input_features, classes).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval(), genome

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local, bounded Aurora neural architecture search")
    parser.add_argument("--generations", type=int, default=4); parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elites", type=int, default=3); parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2400); parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--mutation-rate", type=float, default=.25)
    parser.add_argument("--min-depth", type=int, default=1); parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"); parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("aurora_results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("aurora_champion.pt"),
                        help="Local path for the best model's weights and DNA.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if (args.generations < 1 or args.population < 2 or not 1 <= args.elites < args.population
            or args.samples < 100 or args.features < 5 or args.batch_size < 2
            or not 0 <= args.mutation_rate <= 1 or not 1 <= args.min_depth <= args.max_depth <= 8):
        raise SystemExit("Invalid bounded search configuration; run --help.")
    random.seed(args.seed); torch.manual_seed(args.seed); rng = random.Random(args.seed); device = choose_device(args.device)
    Aurora.info(f"Aurora is local-only · device: {device.type} · synthetic data: {args.samples:,} rows")
    train, validation, classes = synthetic_dataset(args.samples, args.features, args.seed)
    evaluator = FitnessEvaluator(device, args.epochs, args.batch_size); engine = EvolutionEngine(rng, args.population, args.elites, args.mutation_rate, args.min_depth, args.max_depth); dashboard = Dashboard(args.generations)
    population, all_scores = engine.initial_population(), []
    for generation in range(1, args.generations + 1):
        Aurora.mutation(f"Compiling and evaluating {len(population)} valid genomes")
        scores = [evaluator.evaluate(g, train, validation, args.features, classes) for g in population]
        all_scores.extend(scores); dashboard.generation(generation, scores); population = engine.next_population(scores)
    best = max(all_scores, key=lambda e: e.fitness)
    save_result(args.output, best, all_scores, args)
    save_checkpoint(args.checkpoint, best, args.features, classes)
    Aurora.success(
        f"Search complete. Champion metadata: {args.output}; weights: {args.checkpoint}. "
        f"{best.genome.describe()}"
    )
if __name__ == "__main__": main()
