"""Experiment configuration for the multimodal conditional active learning problem.

This module exposes the standardised interface expected by
``scripts/run_experiments.py`` so that the multimodal conditional problem can be
selected with ``--example multimodal_conditional``.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from active_learning_utils import (
    create_shared_multimodal_data,
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
    "random",
    "sbal_mdn_epistemic_variance",
    "sbal_mi_lb",
    "bait",
    "coreset",
}

# ------------------------------------------------------------------
# CLI extensions
# ------------------------------------------------------------------

def register_args(parser):
    """Add example-specific CLI arguments."""
    g = parser.add_argument_group("multimodal-conditional options")

    # Distribution parameters
    g.add_argument("--input-dim", type=int, default=10)
    g.add_argument("--output-dim", type=int, default=16)
    g.add_argument("--latent-dim", type=int, default=4)
    g.add_argument("--num-components", type=int, default=3)
    g.add_argument("--rff-features", type=int, default=128)
    g.add_argument("--c-scale", type=float, default=10.0)
    g.add_argument("--dist-seed", type=int, default=42)
    g.add_argument("--manifold-seed", type=int, default=1)

    # Structured mixing weight parameters
    g.add_argument("--mixing-mode", default="structured",
                   choices=["random", "structured"])
    g.add_argument("--transition-sharpness", type=float, default=8.0)
    g.add_argument("--transition-radius", type=float, default=1.3)
    g.add_argument("--angular-sharpness", type=float, default=2.0)
    g.add_argument("--logit-scale", type=float, default=3.0)

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

    # Override common defaults to match the notebook settings
    parser.set_defaults(
        al_iters=20,
        query_batch_size=50,
        acquisition_batch_size=256,
        training_batch_size=128,
        hidden_features=128,
        depth=2,
        num_mixtures=5,
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
    return create_shared_multimodal_data(
        seed=seed,
        candidate_sample_count=args.candidate_sample_count,
        test_sample_count=args.test_sample_count,
        initial_sample_count=args.initial_sample_count,
        d=args.input_dim,
        m=args.output_dim,
        L=args.latent_dim,
        K=args.num_components,
        p=args.rff_features,
        tau=1.0,
        alpha=0.0,
        c_scale=args.c_scale,
        dist_seed=args.dist_seed,
        manifold_seed=args.manifold_seed,
        mixing_mode=args.mixing_mode,
        transition_sharpness=args.transition_sharpness,
        transition_radius=args.transition_radius,
        angular_sharpness=args.angular_sharpness,
        logit_scale=args.logit_scale,
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def get_default_model_config(args):
    return get_default_mdn_config(
        out_dim=args.output_dim,
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
