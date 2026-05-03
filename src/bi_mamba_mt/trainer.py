"""Backward-compat shim: re-export :mod:`mt_base.trainer`."""

from mt_base.trainer import *  # noqa: F401,F403
from mt_base.trainer import (  # noqa: F401
    EMA,
    TrainConfig,
    build_optimizer,
    cosine_lr,
    train,
)
