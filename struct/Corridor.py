from enum import Enum

print("✅ Loaded updated Corridor class from:", __file__)

class Orientation(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class Corridor:
    def __init__(
        self,
        x: float = None,
        y: float = None,
        width: float = None,
        height: float = None,
        orientation: Orientation = None,
        segments: list[tuple[float, float, float, float]] = None,
    ):
        """
        Represents either:
          - a single straight corridor (x, y, width, height, orientation), or
          - a rectilinear corridor made of multiple rectangular segments.
        segments: list of (x, y, width, height)
        """
        if segments is not None:
            # rectilinear corridor mode
            self.segments = segments
            self.x = None
            self.y = None
            self.width = None
            self.height = None
            self.orientation = None
        else:
            # straight corridor mode
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.orientation = orientation
            self.segments = None

    def __repr__(self):
        if self.segments:
            return f"Corridor(segments={self.segments})"
        return f"Corridor(x={self.x}, y={self.y}, w={self.width}, h={self.height}, orient={self.orientation})"
