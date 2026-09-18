"""Input source contract."""

from abc import ABC, abstractmethod
from typing import Any


class InputSource(ABC):
    @abstractmethod
    def read(self) -> Any:
        """Read one item, or raise a useful error when input is unavailable."""
