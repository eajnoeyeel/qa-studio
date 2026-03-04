"""Common repository primitives."""
import uuid
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

from ...models.database import Base

T = TypeVar("T", bound=Base)


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository."""

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        pass

    @abstractmethod
    def get_all(self, **filters) -> List[T]:
        pass

    @abstractmethod
    def create(self, entity: T) -> T:
        pass

    @abstractmethod
    def update(self, id: str, **kwargs) -> Optional[T]:
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        pass
