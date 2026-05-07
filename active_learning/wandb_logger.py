"""Optional wandb integration for active learning experiments."""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

WANDB_DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "jaxmix-active-learning")


class WandbLogger:
    """Thin wrapper for optional wandb logging in active learning experiments.

    wandb calls are guarded behind ``_WANDB_AVAILABLE`` and ``self.enabled``
    so that code works identically when wandb is not installed.

    Logging strategy:
    - **AL metrics** (``al/*``) are logged via ``run.log`` with ``al/round``
      as the custom x-axis.  Each round produces exactly one history row.
    - **Efficiency metrics** (``efficiency/*``) are logged with
      ``efficiency/train_size`` as the x-axis (test NLL vs training set size).
    - **Training curves** are accumulated in memory and uploaded as a
      ``wandb.Table`` when ``finish()`` is called.  This avoids flooding
      the run history with ~300 rows per round, which causes wandb to
      downsample and drop the sparse ``al/*`` rows.

    Parameters
    ----------
    config : dict
        Flat dict of hyperparameters. Becomes ``wandb.config`` (filterable on
        the dashboard).
    enabled : bool
        Set ``False`` to disable logging even when wandb is installed.
    """

    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        self.enabled = enabled and _WANDB_AVAILABLE
        self.config = config
        self.run = None
        self._global_train_step = 0
        self._training_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_run(self, **kwargs) -> None:
        """Call ``wandb.init`` with the stored config.

        Extra *kwargs* (``entity``, ``project``, ``group``, ``name``, ``tags``,
        ``mode``, etc.) are forwarded directly. Defaults come from
        ``WANDB_ENTITY`` and ``WANDB_PROJECT`` unless overridden here.
        """
        if not self.enabled:
            return
        defaults = {"entity": WANDB_DEFAULT_ENTITY, "project": WANDB_DEFAULT_PROJECT}
        self.run = wandb.init(config=self.config, **{**defaults, **kwargs})

        wandb.define_metric("al/round", hidden=True)
        wandb.define_metric("al/*", step_metric="al/round")
        wandb.define_metric("efficiency/train_size", hidden=True)
        wandb.define_metric("efficiency/*", step_metric="efficiency/train_size")

    def finish(self) -> None:
        """Upload training curves as a Table, then finish the run."""
        if self.enabled and self.run is not None:
            self._upload_training_table()
            self.run.finish()
            self.run = None

    # ------------------------------------------------------------------
    # Per-round logging
    # ------------------------------------------------------------------

    def log_round(self, summary, trainer=None) -> None:
        """Log metrics for a single active-learning round.

        Parameters
        ----------
        summary : ActiveLearningRoundSummary
        trainer : optional trainer with ``loss_log`` / ``grad_norm_log``
        """
        if not self.enabled:
            return

        # Accumulate training curve data (uploaded as Table on finish)
        if trainer is not None:
            self._accumulate_training_curve(trainer, round_idx=summary.round_idx)

        # Log AL round metrics
        self.run.log({
            "al/round": summary.round_idx,
            "al/test_nll": summary.test_loss,
            "al/train_size": summary.train_size,
            "al/pool_size": summary.pool_size,
            "al/mean_acquisition_score": summary.mean_acquisition_score,
            "al/max_acquisition_score": summary.max_acquisition_score,
        }, commit=True)

        # Log efficiency curve (test NLL vs training set size)
        self.run.log({
            "efficiency/train_size": summary.train_size,
            "efficiency/test_nll": summary.test_loss,
        }, commit=True)

    # ------------------------------------------------------------------
    # Training-curve helpers
    # ------------------------------------------------------------------

    def _accumulate_training_curve(self, trainer, round_idx: int = -1) -> None:
        """Buffer training loss/grad-norm for later upload as a Table."""
        loss_log = getattr(trainer, "loss_log", None)
        grad_norm_log = getattr(trainer, "grad_norm_log", None)
        if not loss_log:
            return

        steps_per_check = getattr(trainer, "steps_per_check", 100)
        for i, loss_val in enumerate(loss_log):
            row = {
                "global_step": self._global_train_step,
                "loss": float(loss_val),
                "al_round": round_idx,
                "step_in_round": i * steps_per_check,
            }
            if grad_norm_log and i < len(grad_norm_log):
                row["grad_norm"] = float(grad_norm_log[i])
            self._training_rows.append(row)
            self._global_train_step += 1

    def _upload_training_table(self) -> None:
        """Upload accumulated training data as a wandb Table."""
        if not self._training_rows:
            return
        columns = ["global_step", "loss", "al_round", "step_in_round", "grad_norm"]
        table = wandb.Table(columns=columns)
        for row in self._training_rows:
            table.add_data(
                row["global_step"],
                row["loss"],
                row["al_round"],
                row["step_in_round"],
                row.get("grad_norm", None),
            )
        self.run.log({"train/curves": table})
        self._training_rows.clear()

    # ------------------------------------------------------------------
    # End-of-run logging
    # ------------------------------------------------------------------

    def log_final(self, final_test_nll: float, state, final_trainer=None, final_round_idx: int = -1) -> None:
        """Log end-of-run summary metrics and a final AL data point.

        Parameters
        ----------
        final_test_nll : float
        state : ActiveLearningState
        final_trainer : optional trainer for the final round's training curve
        final_round_idx : int
            Round index for the final data point (e.g. ``num_rounds``).
        """
        if not self.enabled:
            return

        if final_trainer is not None:
            self._accumulate_training_curve(final_trainer, round_idx=final_round_idx)

        train_size = int(state.train_inputs.shape[0])
        self.run.log({
            "al/round": final_round_idx,
            "al/test_nll": final_test_nll,
            "al/train_size": train_size,
        }, commit=True)

        self.run.log({
            "efficiency/train_size": train_size,
            "efficiency/test_nll": final_test_nll,
        }, commit=True)

        self.run.summary["final_test_nll"] = final_test_nll
        self.run.summary["final_train_size"] = train_size

    def log_figure(self, key: str, fig) -> None:
        """Log a matplotlib figure as a ``wandb.Image``."""
        if not self.enabled:
            return
        self.run.log({key: wandb.Image(fig)})
        plt.close(fig)
