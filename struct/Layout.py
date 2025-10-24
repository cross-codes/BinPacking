from dataclasses import dataclass
from struct.PlacedRoom import PlacedRoom
from struct.Corridor import Corridor
from struct.RoomSpec import RoomSpec 


@dataclass
class Layout:
    placed: list[PlacedRoom]
    corridor: Corridor
    plot_w: float
    plot_h: float
    rooms_area: float
    placed_count: int
    label: str
    unplaced: list[RoomSpec]
