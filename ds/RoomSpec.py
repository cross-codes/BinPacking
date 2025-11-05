class RoomSpec:
    def __init__(self, name: str, w: float, h: float, can_rotate: bool = True):
        self.name: str = name
        self.w: float = float(w)
        self.h: float = float(h)
        self.can_rotate: bool = can_rotate
