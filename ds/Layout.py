from ds.Corridor import Corridor
from ds.PlacedRoom import PlacedRoom


class Layout:
    def __init__(
        self,
        placed: list[PlacedRoom],
        unplaced: list[str],
        plot_w: float,
        plot_h: float,
        corridor: Corridor | None = None,
        backbone: set[tuple[int, int]] | None = None,
    ):
        self.placed: list[PlacedRoom] = placed
        self.unplaced_names: list[str] = unplaced
        self.plot_w: float = plot_w
        self.plot_h: float = plot_h
        self.corridor: Corridor | None = corridor
        self.backbone: set[tuple[int, int]] | None = backbone
        self.placed_count: int = len(placed)
        self.entrance_pos: tuple[float, float] | None = None
        self.free_boundary_segments: dict[str, list[tuple[float, float]]] | None = None
