import datetime
import logging
import os
from slime.utils.misc import SingletonMeta

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

__all__ = ["_TensorboardAdapter"]

logger = logging.getLogger(__name__)


class _TensorboardAdapter(metaclass=SingletonMeta):
    _writers = None
    _tensorboard_dir = None

    """
    # Usage example: This will return the same instance every rank
    # tb = _TensorboardAdapter(args)  # Initialize on first call
    # tb.log({"Loss": 0.1}, step=1)

    # In other files:
    # from tensorboard_utils import _TensorboardAdapter
    # tb = _TensorboardAdapter(args)  # No parameters needed to get existing instance
    # tb.log({"Accuracy": 0.9}, step=1)
    """

    def __init__(self, args):
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        if tb_project_name is not None or os.environ.get("TENSORBOARD_DIR", None):
            if tb_project_name is not None and tb_experiment_name is None:
                tb_experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._initialize(tb_project_name, tb_experiment_name)
        else:
            raise ValueError("tb_project_name and tb_experiment_name, or TENSORBOARD_DIR are required")

    def _initialize(self, tb_project_name, tb_experiment_name):
        """Actual initialization logic"""
        # Get tensorboard directory from environment variable or use default path
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{tb_project_name}/{tb_experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        self._tensorboard_dir = tensorboard_dir
        self._writers = {}

    def _stream_for_key(self, key: str) -> str:
        if key.startswith("train/critic-"):
            return "critic"
        if key.startswith("train/"):
            return "actor"
        if key.startswith("rollout/") or key.startswith("perf/"):
            return "rollout"
        if key.startswith("reward/"):
            return "reward"
        return "misc"

    def _get_writer(self, stream: str):
        writer = self._writers.get(stream)
        if writer is not None:
            return writer

        writer_dir = os.path.join(self._tensorboard_dir, stream)
        os.makedirs(writer_dir, exist_ok=True)
        logger.info("Saving tensorboard %s log to %s.", stream, writer_dir)
        writer = SummaryWriter(writer_dir)
        self._writers[stream] = writer
        return writer

    def log(self, data, step):
        """Log data to tensorboard

        Args:
            data (dict): Dictionary containing metric names and values
            step (int): Current step/epoch number
        """
        for key, value in data.items():
            writer = self._get_writer(self._stream_for_key(key))
            writer.add_scalar(key, value, step)

    def finish(self):
        """Close the tensorboard writer"""
        for writer in (self._writers or {}).values():
            writer.close()
