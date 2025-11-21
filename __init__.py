"""
Tiny Micro Torch - A minimal, educational automatic differentiation engine and neural network library.
"""

# Import main components to make them available at the package level
from .autograd import *
from .nn import *
from .optim import *
from .data import *
from .reporters import *

__version__ = "0.1.0"
