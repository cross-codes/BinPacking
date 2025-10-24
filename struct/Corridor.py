from enum import Enum
from dataclasses import dataclass


class Orientation(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@dataclass
class Corridor:
    x: float
    y: float
    width: float
    height: float
    orientation: Orientation
