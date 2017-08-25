try:
    from ._version import __version__
except(ImportError):
    pass

from . import fitting
from . import predict
from . import train
from . import train_gpu
from . import testing
from . import utils
