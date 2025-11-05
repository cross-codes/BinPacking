from ds.Orientation import Orientation


class Corridor:
    def __init__(
        self, x: float, y: float, w: float, h: float, orientation: Orientation
    ):
        self.x: float = x
        self.y: float = y
        self.w: float = w
        self.h: float = h
        self.orientation: Orientation = orientation
