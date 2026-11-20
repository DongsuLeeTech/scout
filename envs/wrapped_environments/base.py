from typing import Any, Dict, Tuple

import numpy as np

Observations = Dict[str, np.ndarray]
NextObservations = Observations
Rewards = Dict[str, np.ndarray]
Terminals = Dict[str, np.ndarray]
Truncations = Dict[str, np.ndarray]
Info = Dict[str, Any]

ResetReturn = Tuple[Observations, Info]
StepReturn = Tuple[NextObservations, Rewards, Terminals, Truncations, Info]


class BaseEnvironment:

    def __init__(self) -> None:
        pass

    def reset(self) -> ResetReturn:
        raise NotImplementedError

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        raise NotImplementedError

    def get_stats(self) -> Dict:
        return {}

    def render(self) -> Any:
        return np.zeros((10, 10, 3), "float32")

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
