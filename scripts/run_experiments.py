#!/usr/bin/env python
"""Run active learning experiments with wandb logging.

This runner dispatches to example-specific code via the ``--example`` flag.
Each example lives under ``examples/<name>/`` and provides an
``experiment_config.py`` module.

Usage examples:
    # Run one example
    python scripts/run_experiments.py --example multimodal_conditional

    # Single acquisition, single scenario
    python scripts/run_experiments.py --example coupled_double_well \
        --acquisition mi_lb --query-scenario pool_based

    # Multiple acquisitions and seeds (comma-separated)
    python scripts/run_experiments.py --example ternary_phases \
        --acquisition random,mi_lb --seed 42,123,456

    # Offline wandb (sync later with `wandb sync`)
    python scripts/run_experiments.py --example multimodal_conditional --wandb-offline

    # Disable wandb entirely
    python scripts/run_experiments.py --example multimodal_conditional --no-wandb

    # Remote execution (nohup)
    nohup python scripts/run_experiments.py --example multimodal_conditional \
        > experiments.log 2>&1 &
"""

import argparse
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_learning.wandb_logger import WandbLogger

import matplotlib
matplotlib.use("Agg")


def _load_example_config(example_name: str):
    """Dynamically import ``examples/<example_name>/experiment_config``."""
    example_dir = _REPO_ROOT / "examples" / example_name
    if not example_dir.is_dir():
        available = sorted(
            p.name for p in (_REPO_ROOT / "examples").iterdir() if p.is_dir()
        )
        raise SystemExit(
            f"Example '{example_name}' not found under examples/.\n"
            f"Available: {available}"
        )

    config_path = example_dir / "experiment_config.py"
    if not config_path.exists():
        raise SystemExit(
            f"Example '{example_name}' has no experiment_config.py.\n"
            f"Please create {config_path} implementing the required interface."
        )

    # Add the example directory to sys.path so that its local imports
    # (e.g. ``from active_learning_utils import ...``) resolve correctly.
    example_dir_str = str(example_dir)
    if example_dir_str not in sys.path:
        sys.path.insert(0, example_dir_str)

    spec = importlib.util.spec_from_file_location(
        f"examples.{example_name}.experiment_config", config_path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_common_parser():
    """Arguments shared across all examples."""
    p = argparse.ArgumentParser(
        description="Run active learning experiments with wandb logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- example selection ---
    p.add_argument("--example", required=True,
                    help="Name of the example directory under examples/ "
                         "(e.g. multimodal_conditional)")

    # --- wandb ---
    p.add_argument("--wandb-entity", default=None,
                    help="wandb team/user entity (ensures collaborators log to the "
                         "same place)")
    p.add_argument("--project", default="jaxmix-active-learning",
                    help="wandb project name")
    p.add_argument("--no-wandb", action="store_true",
                    help="disable wandb logging entirely")
    p.add_argument("--wandb-offline", action="store_true",
                    help="run wandb in offline mode (sync later)")
    p.add_argument("--run-suffix", default=None,
                    help="suffix appended to wandb run name (e.g. --run-suffix v2)")

    # --- experiment grid ---
    p.add_argument("--seed", nargs="+", default=["42"],
                    help="comma-separated seed(s) (e.g. --seed 1,2,3 or --seed 1, 2, 3)")
    p.add_argument("--acquisition", nargs="+",
                    default=["random,mdn_epistemic_variance,mi_lb"],
                    help="comma-separated acquisition function(s) "
                         "(e.g. --acquisition random,mi_lb)")
    p.add_argument("--query-scenario", nargs="+", default=None,
                    help="query scenario(s) — defaults to all supported by the example")

    # --- AL loop ---
    p.add_argument("--al-iters", type=int, default=8)
    p.add_argument("--query-batch-size", type=int, default=50)
    p.add_argument("--acquisition-batch-size", type=int, default=512)
    p.add_argument("--selection-strategy", default="top_k",
                    choices=["top_k", "sbal", "maxdist"],
                    help="selection strategy for pool-based AL")

    # --- model ---
    p.add_argument("--ensemble-size", type=int, default=12)
    p.add_argument("--num-mixtures", type=int, default=8)
    p.add_argument("--hidden-features", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=30_000)
    p.add_argument("--training-batch-size", type=int, default=512)

    # --- data ---
    p.add_argument("--candidate-sample-count", type=int, default=2000)
    p.add_argument("--test-sample-count", type=int, default=2000)
    p.add_argument("--initial-sample-count", type=int, default=50)

    return p


def _resolve_selection_strategy(acq_name, default_strategy):
    """Determine the actual selection strategy from an acquisition name."""
    if acq_name.startswith("sbal_"):
        return "sbal"
    elif acq_name.startswith("maxdist_"):
        return "maxdist"
    return default_strategy


def _build_run_config(args, seed, scenario, acq_name):
    """Build the flat config dict that becomes wandb.config.

    Logs all CLI args (including example-specific ones) so nothing is missed.
    """
    # Start with all parsed args to capture example-specific params
    config = {k: v for k, v in vars(args).items() if k not in ("seed", "acquisition")}
    # Override with per-run values
    config.update({
        "seed": seed,
        "acquisition_name": acq_name,
        "query_scenario": scenario,
        "selection_strategy": _resolve_selection_strategy(acq_name, args.selection_strategy),
    })
    return config


def main():
    # Two-pass parse: first grab --example so we can load the config and
    # let it register extra args, then re-parse with the full parser.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--example", required=True)
    pre_args, _ = pre_parser.parse_known_args()

    example_cfg = _load_example_config(pre_args.example)

    parser = _build_common_parser()
    if hasattr(example_cfg, "register_args"):
        example_cfg.register_args(parser)
    args = parser.parse_args()

    args.seed = [int(s.strip()) for s in ",".join(args.seed).split(",") if s.strip()]
    args.acquisition = [a.strip() for a in ",".join(args.acquisition).split(",") if a.strip()]

    scenarios = args.query_scenario
    if scenarios is None:
        scenarios = list(example_cfg.SUPPORTED_SCENARIOS)
    unsupported = set(scenarios) - set(example_cfg.SUPPORTED_SCENARIOS)
    if unsupported:
        raise SystemExit(
            f"Scenarios {unsupported} not supported by '{args.example}'. "
            f"Supported: {example_cfg.SUPPORTED_SCENARIOS}"
        )

    non_diff = getattr(example_cfg, "NON_DIFFERENTIABLE_ACQUISITIONS", set())

    for seed in args.seed:
        print(f"\n{'='*60}")
        print(f"Example: {args.example}  |  Seed: {seed}")
        print(f"{'='*60}")

        data = example_cfg.create_data(args, seed)
        model_config = example_cfg.get_default_model_config(args)
        trainer_factory = example_cfg.make_trainer_factory(model_config, args)

        all_results = {}

        for scenario in scenarios:
            acquisitions = list(args.acquisition)
            if scenario == "gradient":
                acquisitions = [a for a in acquisitions if a not in non_diff]

            for acq_name in acquisitions:
                run_config = _build_run_config(args, seed, scenario, acq_name)
                label = (f"{acq_name}" if scenario == "pool_based"
                         else f"gradient ({acq_name})")

                print(f"\n--- Running: {label} ---")

                logger = WandbLogger(run_config, enabled=not args.no_wandb)
                wandb_kwargs = {
                    "project": args.project,
                    "group": f"{args.example}/seed-{seed}",
                    "name": f"{scenario}/{acq_name}/seed_{seed}" + (f"/{args.run_suffix}" if args.run_suffix else ""),
                    "tags": [args.example, scenario, acq_name],
                }
                if args.wandb_entity:
                    wandb_kwargs["entity"] = args.wandb_entity
                if args.wandb_offline:
                    wandb_kwargs["mode"] = "offline"
                logger.init_run(**wandb_kwargs)

                result = example_cfg.run_experiment(
                    scenario, trainer_factory, data, acq_name, args, logger,
                )

                # Log summary figure
                fig = example_cfg.make_single_run_figure(result, title=label)
                logger.log_figure("plots/summary", fig)

                logger.finish()
                all_results[label] = result
                print(f"Done: {label} -- final NLL = {result['final_test_nll']:.4f}")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
