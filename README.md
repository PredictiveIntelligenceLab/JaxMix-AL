# MI-LB: Active Learning for Multimodal Continuous Outputs

Code release accompanying the paper. Implements the **Mutual Information Lower
Bound (MI-LB)** acquisition function for active learning with mixture-density
networks, plus the three benchmarks reported in Section 5 (synthetic
multimodal conditional, coupled double-well system, ternary phase-competition).

## Repository layout

```
.
├── jaxmix/                    # JAX/Flax model and training utilities
│   ├── archs.py               #   MDN architectures
│   ├── trainers.py            #   training loop primitives
│   ├── data_loaders.py
│   └── utils.py
├── active_learning/           # Active-learning framework
│   ├── loop.py                #   pool-based AL loop
│   ├── state.py               #   labelled / unlabelled set bookkeeping
│   ├── history.py             #   round-by-round logging
│   ├── wandb_logger.py        #   W&B integration
│   ├── acquisitions/          #   Random, Variance, MI-LB, BAIT, Core-Set
│   ├── selection/             #   top-k, SBAL, MaxDist
│   └── query_scenarios/       #   pool-based scenario
├── examples/                  # The three benchmarks from Section 5
│   ├── multimodal_conditional/
│   ├── coupled_double_well/
│   └── ternary_phases/
└── scripts/
    └── run_experiments.py     # CLI runner used to launch the W&B sweeps
```

Each `examples/<benchmark>/` directory contains:

| File | Purpose |
|---|---|
| `<benchmark>.py` | data-generating simulator |
| `experiment_config.py` | per-benchmark hyperparameter defaults |
| `active_learning_utils.py` | benchmark-specific trainer / model factories |
| `active_learning.ipynb` | end-to-end AL run for one (acquisition, seed) pair |
| `mdn_baseline.ipynb` | offline `K = 1` vs `K = K_MDN` mixture-head ablation |

Figure-rendering scripts (which pull finished W&B runs and aggregate across
seeds) are not included in this code release; the rendered figures appear
in the paper itself.

## Installation

The code targets Python 3.10+. Install dependencies into a fresh virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A GPU-enabled JAX install is recommended for the active-learning loops
(see [JAX install instructions](https://github.com/google/jax#installation)
to match your CUDA version).

## Reproducing the paper

The active-learning experiments optionally log to
[Weights & Biases](https://wandb.ai/). To reproduce:

1. **(Optional) Configure W&B** — log in (`wandb login`) and set your entity
   via the `WANDB_ENTITY` environment variable. To run without any W&B
   logging, set `WANDB_MODE=disabled`.

2. **Launch the AL runs.** From each `examples/<benchmark>/` directory, run
   the `active_learning.ipynb` notebook for every combination of acquisition
   function and seed reported in the paper:

   - acquisitions: `random`, `mdn_epistemic_variance`, `mi_lb`, `bait`, `coreset`
   - SBAL variants: `sbal_mdn_epistemic_variance`, `sbal_mi_lb`
   - MaxDist variant: `maxdist_mi_lb`
   - seeds: `0, 1, 2, 3, 4`

   Each run tags itself with the benchmark name (`multimodal_conditional`,
   `coupled_double_well`, `ternary_phases`) and (when W&B is enabled) gets
   logged as a separate W&B run with the configured acquisition and seed.

   (Alternatively, `scripts/run_experiments.py` exposes the same workflow
   from the command line.)

3. **Mixture-head ablations.** Each `mdn_baseline.ipynb` re-runs the
   `K = 1` vs `K = K_MDN` offline ablation reported in the appendix.

## Benchmark hyperparameters at a glance

| | Multimodal | Double-well | Ternary |
|---|---|---|---|
| Input dim | 10 | 7 | 8 |
| Output dim | 16 | 20 | 1 |
| MDN components ($K$) | 5 | 8 | 4 |
| Ensemble size | 8 | 8 | 8 |
| Pool size | 50,000 | 50,000 | 50,000 |
| Initial labelled | 100 | 100 | 100 |
| AL rounds × batch | 20 × 50 | 20 × 50 | 30 × 15 |
| Total labels | 1,100 | 1,100 | 550 |
| Optimiser | AdamW | AdamW | AdamW |
| Peak LR / weight decay | 5e-4 / 1e-2 | 5e-4 / 1e-2 | 2e-4 / 5e-2 |

Full hyperparameter tables and training-schedule details are in Appendix C
of the paper.

## Compute

All experiments were run on a single NVIDIA RTX A6000 (48 GB VRAM) per
active-learning run. Total compute across the three benchmarks (8 acquisition
× 5 seeds = 40 runs per benchmark, 120 runs total) is approximately
25 GPU-hours; see Appendix C.4 for per-method wall-clock times. GPU memory
is not a binding constraint at this model size.
