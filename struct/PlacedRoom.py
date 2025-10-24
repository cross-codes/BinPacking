from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class PlacedRoom:
    name: str
    width: float
    height: float
    x: float
    y: float
    rotated: bool
    side: Side
