"""Experiment configuration for the coupled double-well active learning problem.

This module exposes the standardised interface expected by
``scripts/run_experiments.py`` so that the double-well problem can be
selected with ``--example coupled_double_well``.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from active_learning_utils import (
    create_shared_double_well_data,
    get_default_mdn_config,
    make_mdn_trainer_factory,
    run_pool_based_active_learning_experiment,
    make_single_run_figure,
    plot_loss_comparison,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

SUPPORTED_SCENARIOS = ["pool_based"]
NON_DIFFERENTIABLE_ACQUISITIONS = {
    "random", "sbal_mdn_epistemic_variance", "sbal_mi_lb", "bait", "coreset",
}

# ------------------------------------------------------------------
# CLI extensions
# ------------------------------------------------------------------

def register_args(parser):
    """Add example-specific CLI arguments."""
    g = parser.add_argument_group("coupled-double-well options")

    # Physics parameters
    g.add_argument("--particles", type=int, default=5)
    g.add_argument("--integration-time", type=float, default=5.0)
    g.add_argument("--dt", type=float, default=0.005)
    g.add_argument("--n-snapshots", type=int, default=4)
    g.add_argument("--sigma-lo", type=float, default=0.3)
    g.add_argument("--sigma-hi", type=float, default=2.0)
    g.add_argument("--kappa-lo", type=float, default=0.0)
    g.add_argument("--kappa-hi", type=float, default=3.0)

    # Adaptive iteration scaling
    g.add_argument("--adaptive-iters", action="store_true", default=True)
    g.add_argument("--no-adaptive-iters", dest="adaptive_iters", action="store_false")
    g.add_argument("--iter-per-sample", type=int, default=10)

    g.add_argument("--sbal-temperature", type=float, default=0.3)

    # Core-Set (Sener & Savarese, ICLR 2018) parameters
    g.add_argument("--coreset-ensemble-member", type=int, default=0,
                   help="ensemble index whose backbone Core-Set uses")
    g.add_argument("--coreset-pool-subsample", type=int, default=0,
                   help="subsample pool to this many candidates before Core-Set "
                        "(set to 0 to disable subsampling)")

    # Override common defaults
    parser.set_defaults(
        al_iters=20,
        query_batch_size=50,
        acquisition_batch_size=256,
        training_batch_size=128,
        hidden_features=128,
        depth=3,
        num_mixtures=8,
        ensemble_size=8,
        n_iter=10_000,
        candidate_sample_count=50_000,
        test_sample_count=2_000,
        initial_sample_count=100,
        acquisition=["random,mdn_epistemic_variance,sbal_mdn_epistemic_variance,mi_lb,sbal_mi_lb,bait,coreset"],
    )


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def create_data(args, seed):
    return create_shared_double_well_data(
        seed=seed,
        candidate_sample_count=args.candidate_sample_count,
        test_sample_count=args.test_sample_count,
        initial_sample_count=args.initial_sample_count,
        P=args.particles,
        T=args.integration_time,
        dt=args.dt,
        n_snapshots=args.n_snapshots,
        sigma_range=(args.sigma_lo, args.sigma_hi),
        kappa_range=(args.kappa_lo, args.kappa_hi),
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def get_default_model_config(args):
    P = args.particles
    out_dim = P if args.n_snapshots == 0 else args.n_snapshots * P
    return get_default_mdn_config(
        out_dim=out_dim,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_mixtures=args.num_mixtures,
        ensemble_size=args.ensemble_size,
    )


def make_trainer_factory(model_config, args):
    return make_mdn_trainer_factory(
        model_config,
        n_iter=args.n_iter,
        batch_size=args.training_batch_size,
        adaptive_iters=args.adaptive_iters,
        iter_per_sample=args.iter_per_sample,
    )


# ------------------------------------------------------------------
# Experiment dispatch
# ------------------------------------------------------------------

def run_experiment(scenario, trainer_factory, data, acq_name, args, logger):
    """Run a single experiment and return a results dict."""
    if scenario == "pool_based":
        return run_pool_based_active_learning_experiment(
            trainer_factory=trainer_factory,
            shared_benchmark=data,
            acquisition_name=acq_name,
            al_iters=args.al_iters,
            query_batch_size=args.query_batch_size,
            acquisition_batch_size=args.acquisition_batch_size,
            selection_strategy=args.selection_strategy,
            sbal_temperature=args.sbal_temperature,
            coreset_ensemble_member=args.coreset_ensemble_member,
            coreset_pool_subsample=(
                args.coreset_pool_subsample if args.coreset_pool_subsample > 0 else None
            ),
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
