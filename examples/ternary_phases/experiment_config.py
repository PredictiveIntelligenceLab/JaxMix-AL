"""Experiment configuration for the ternary phases active learning problem."""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from active_learning_utils import (
    create_shared_ternary_data,
    get_default_mdn_config,
    make_mdn_trainer_factory,
    run_pool_based_active_learning_experiment,
    make_single_run_figure,
    plot_loss_comparison,
)

SUPPORTED_SCENARIOS = ["pool_based"]
NON_DIFFERENTIABLE_ACQUISITIONS = {
    "random", "sbal_mdn_epistemic_variance", "sbal_mi_lb", "bait", "coreset",
}

def register_args(parser):
    g = parser.add_argument_group("ternary-phases options")

    g.add_argument("--n-phases", type=int, default=4)
    g.add_argument("--tau-G", type=float, default=0.08,
                   help="Softmin temperature.  Lower = sharper boundaries.")
    g.add_argument("--n-proc-params", type=int, default=6)
    g.add_argument("--c-mu-scale", type=float, default=6.0,
                   help="Scale of per-phase response coefficients.")
    g.add_argument("--system-seed", type=int, default=12)

    g.add_argument("--adaptive-iters", action="store_true", default=True)
    g.add_argument("--no-adaptive-iters", dest="adaptive_iters", action="store_false")
    g.add_argument("--iter-per-sample", type=int, default=200)

    g.add_argument("--sbal-temperature", type=float, default=0.3)

    # Core-Set (Sener & Savarese, ICLR 2018) parameters
    g.add_argument("--coreset-ensemble-member", type=int, default=0,
                   help="ensemble index whose backbone Core-Set uses")
    g.add_argument("--coreset-pool-subsample", type=int, default=0,
                   help="subsample pool to this many candidates before Core-Set "
                        "(set to 0 to disable subsampling)")

    parser.set_defaults(
        al_iters=30,
        query_batch_size=15,
        acquisition_batch_size=256,
        training_batch_size=64,
        hidden_features=64,
        depth=2,
        num_mixtures=4,
        ensemble_size=8,
        n_iter=40_000,
        candidate_sample_count=50_000,
        test_sample_count=2_000,
        initial_sample_count=100,
        acquisition=["random,mdn_epistemic_variance,sbal_mdn_epistemic_variance,mi_lb,sbal_mi_lb,bait,coreset"],
    )


def create_data(args, seed):
    return create_shared_ternary_data(
        seed=seed,
        system_seed=args.system_seed,
        candidate_sample_count=args.candidate_sample_count,
        test_sample_count=args.test_sample_count,
        initial_sample_count=args.initial_sample_count,
        n_phases=args.n_phases,
        tau_G=args.tau_G,
        n_proc_params=args.n_proc_params,
        c_mu_scale=args.c_mu_scale,
    )


def get_default_model_config(args):
    return get_default_mdn_config(
        out_dim=1,
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


def run_experiment(scenario, trainer_factory, data, acq_name, args, logger):
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
    raise ValueError(f"Unsupported scenario: {scenario}")
