import logging
import re
import sys
from typing import Sequence, TextIO, Union

import numpy as np
import wandb
from tensorboardX import SummaryWriter


class _Logger:
    class LoggerWriter:
        def __init__(self, console: TextIO, file: TextIO):
            self.console = console
            self.file = file

        def write(self, message):
            self.console.write(message)
            self.file.write(message)

        def flush(self):
            self.console.flush()
            self.file.flush()

    def __init__(self):
        self.logger = self._init_logger()
        self.wandb = False
        self.tb = False

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger()
        logger.setLevel("INFO")
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def setup_output_file(self, filepath):
        log_file = open(filepath, "a")
        sys.stdout = _Logger.LoggerWriter(sys.stdout, log_file)
        sys.stderr = _Logger.LoggerWriter(sys.stderr, log_file)
        self.logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setup_wandb(self, project: str = None, entity: str = None, group: str = None, job_type: str = None, dir: str = None, config: dict = None, name: str = None, **kwargs):
        wandb.init(project=project, entity=entity, group=group, job_type=job_type, dir=dir, config=config, name=name, **kwargs)
        self.wandb = True

    def setup_tensorboard(self, log_dir: str):
        self.tb_writter = SummaryWriter(log_dir)
        self.tb = True

    def log_metrics(self, metrics, step: int = None):
        if self.wandb:
            wandb.log(metrics, step=step)
        if self.tb:
            for k, v in metrics.items():
                self.tb_writter.add_scalar(k, v, global_step=step)

    def __del__(self):
        if self.wandb:
            wandb.finish()
        if self.tb:
            self.tb_writter.close()


_logger = _Logger()


def info(msg, *args, **kwargs):
    _logger.logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    _logger.logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _logger.logger.error(msg, *args, **kwargs)


def setup_output_file(filepath):
    _logger.setup_output_file(filepath)


def setup_wandb(project: str = None, entity: str = None, group: str = None, job_type: str = None, dir: str = None, config: dict = None, name: str = None, **kwargs):
    _logger.setup_wandb(project=project, entity=entity, group=group, job_type=job_type, dir=dir, config=config, name=name, **kwargs)


def setup_tensorboard(log_dir: str):
    _logger.setup_tensorboard(log_dir)


def log_metrics(title, metrics, step: int = None, filtered_keys: Sequence[Union[str, re.Pattern]] = None):
    log_str = """
{}
========================================
{}
========================================
    """
    metric_values = {k: np.mean(v) for k, v in sorted(metrics.items())}
    _logger.log_metrics(metric_values, step=step)

    if filtered_keys is not None:
        filtered_rules = [re.compile(k) if isinstance(k, str) else k for k in filtered_keys]
        metric_values = {k: v for k, v in metric_values.items() if not any(rule.match(k) for rule in filtered_rules)}

    metrics_str = "\n".join([f"{k:<30}  {f'{v:.4f}':>8}" for k, v in metric_values.items()])
    return log_str.format(title, metrics_str)
