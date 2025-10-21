from enum import Enum
from typing import TypeAlias

class ChameleonModelName(str, Enum):
    LENOY_ANOLE_7B_V01 = "leloy/Anole-7b-v0.1-hf"

class FakeModelName(str, Enum):
    """Fake model for testing."""
    FAKE = "fake"

AllModelEnum: TypeAlias = (
    ChameleonModelName 
    | FakeModelName
)