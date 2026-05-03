"""Backward-compat shim: re-export :mod:`mt_base.evaluator`."""

from mt_base.evaluator import *  # noqa: F401,F403
from mt_base.evaluator import (  # noqa: F401
    DEFAULT_LENGTH_BUCKETS,
    EvalResult,
    evaluate,
)
