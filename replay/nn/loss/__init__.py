from .base import LossProto
from .bce import BCE, BCESampled
from .ce import CE, CESampled
from .login_ce import LogInCE, LogInCESampled
from .logout_ce import LogOutCE
from .fourier import InBatchFourierLoss

LogOutCESampled = CE

__all__ = [
    "BCE",
    "CE",
    "BCESampled",
    "CESampled",
    "LogInCE",
    "LogInCESampled",
    "LogOutCE",
    "LogOutCESampled",
    "LossProto",
    "InBatchFourierLoss",
]
