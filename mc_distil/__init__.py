"""
MC-Distil: Knowledge Distillation Library
"""

__version__ = "0.1.0"
__author__ = "MC-Distil Team"

from . import models
from . import training
from . import utils
from . import data

__all__ = ['models', 'training', 'utils', 'data']