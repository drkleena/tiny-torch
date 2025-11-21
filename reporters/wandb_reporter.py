"""
Weights & Biases (wandb) integration for training reporting.
"""

from typing import Optional, Dict, Any
import wandb


class TrainingReporter:
    """
    A training reporter that logs metrics to Weights & Biases.

    This class provides hooks for logging training metrics, model configuration,
    and evaluation results to wandb for experiment tracking and visualization.

    Example:
        >>> reporter = TrainingReporter(
        ...     project="mnist-mlp",
        ...     config={"lr": 0.1, "batch_size": 1000, "epochs": 10}
        ... )
        >>> reporter.log_batch(loss=0.5, batch_idx=0, epoch=0)
        >>> reporter.log_epoch(avg_loss=0.45, epoch=0)
        >>> reporter.log_evaluation(accuracy=0.95, split="test")
        >>> reporter.finish()
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        enabled: bool = True,
        **kwargs
    ):
        """
        Initialize the training reporter.

        Args:
            project: wandb project name
            name: Optional run name. If None, wandb generates one
            config: Dictionary of hyperparameters and config values to log
            tags: List of tags for this run
            notes: Optional notes about this run
            enabled: If False, all logging is disabled (useful for debugging)
            **kwargs: Additional arguments passed to wandb.init()
        """
        self.enabled = enabled
        self.project = project

        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                **kwargs
            )
        else:
            self.run = None

    def log_batch(
        self,
        loss: float,
        batch_idx: int,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log metrics for a single training batch.

        Args:
            loss: Training loss for this batch
            batch_idx: Index of the batch within the epoch
            epoch: Current epoch number
            metrics: Optional additional metrics to log
        """
        if not self.enabled:
            return

        log_dict = {
            "batch/loss": loss,
            "batch/idx": batch_idx,
            "batch/epoch": epoch,
        }

        if metrics:
            for key, value in metrics.items():
                log_dict[f"batch/{key}"] = value

        wandb.log(log_dict)

    def log_epoch(
        self,
        avg_loss: float,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log metrics for a completed epoch.

        Args:
            avg_loss: Average loss across all batches in the epoch
            epoch: Epoch number
            metrics: Optional additional metrics to log
        """
        if not self.enabled:
            return

        log_dict = {
            "epoch": epoch,
            "epoch/avg_loss": avg_loss,
        }

        if metrics:
            for key, value in metrics.items():
                log_dict[f"epoch/{key}"] = value

        wandb.log(log_dict)

    def log_evaluation(
        self,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        split: str = "test",
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log evaluation metrics.

        Args:
            accuracy: Evaluation accuracy (0-1)
            loss: Evaluation loss
            split: Data split name (e.g., "test", "val", "train")
            metrics: Optional additional metrics to log
        """
        if not self.enabled:
            return

        log_dict = {}

        if accuracy is not None:
            log_dict[f"{split}/accuracy"] = accuracy

        if loss is not None:
            log_dict[f"{split}/loss"] = loss

        if metrics:
            for key, value in metrics.items():
                log_dict[f"{split}/{key}"] = value

        if log_dict:
            wandb.log(log_dict)

    def log_metrics(self, metrics: Dict[str, Any], prefix: Optional[str] = None):
        """
        Log arbitrary metrics.

        Args:
            metrics: Dictionary of metric names to values
            prefix: Optional prefix to add to all metric names
        """
        if not self.enabled:
            return

        log_dict = {}
        for key, value in metrics.items():
            if prefix:
                log_dict[f"{prefix}/{key}"] = value
            else:
                log_dict[key] = value

        wandb.log(log_dict)

    def log_images(
        self,
        images: list,
        caption: Optional[str] = None,
        prefix: str = "images"
    ):
        """
        Log images to wandb.

        Args:
            images: List of wandb.Image objects or numpy arrays
            caption: Optional caption for the images
            prefix: Prefix for the logged images (default: "images")
        """
        if not self.enabled:
            return

        # Convert numpy arrays to wandb.Image if needed
        wandb_images = []
        for img in images:
            if isinstance(img, wandb.Image):
                wandb_images.append(img)
            else:
                # Assume it's a numpy array
                wandb_images.append(wandb.Image(img, caption=caption))

        wandb.log({prefix: wandb_images})

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch a model's parameters and gradients.

        Note: This may not work perfectly with custom autograd engines.
        Use with caution.

        Args:
            model: Model to watch
            log: What to log - "gradients", "parameters", "all", or None
            log_freq: Logging frequency
        """
        if not self.enabled:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"Warning: Could not watch model: {e}")

    def finish(self):
        """
        Finish the wandb run.

        Call this at the end of training to properly close the wandb session.
        """
        if self.enabled and self.run is not None:
            wandb.finish()

    def __enter__(self):
        """Enable context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure wandb is finished when exiting context."""
        self.finish()
        return False
