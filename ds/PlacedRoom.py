class PlacedRoom:
    def __init__(
        self, name: str, x: float, y: float, w: float, h: float, rotated: bool
    ):
        self.name: str = name
        self.x: float = float(x)
        self.y: float = float(y)
        self.w: float = float(w)
        self.h: float = float(h)
        self.rotated: bool = bool(rotated)
