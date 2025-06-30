from typing import Any, Protocol

import numpy as np


class Scaler(Protocol):
    def __call__(
        self, data: np.ndarray, context: Any = None
    ) -> tuple[np.ndarray, Any]: ...
    def unscale(self, data: np.ndarray, context: Any) -> np.ndarray: ...
