from typing import override
from constants import *
from algorithm.AbstractAlgorithm import AbstractAlgorithm
from ds.RoomSpec import RoomSpec
from ds.Layout import Layout
from ds.Side import Side
from ds.PlacedRoom import PlacedRoom
from ds.Corridor import Corridor
from ds.Orientation import Orientation


class SingleCorridorAlgorithm(AbstractAlgorithm):
    def __init__(self, plot_w: float, plot_h: float, rooms: list[RoomSpec]):
        super().__init__(plot_w, plot_h, rooms)
        self.CORRIDOR_WIDTH = 2.0
        self.AREA_FRACTION_LIMIT = 0.9

    @override
    def generate(self) -> list[Layout]:
        if self.W <= 0 or self.H <= 0:
            return []

        cap = self.AREA_FRACTION_LIMIT * self.W * self.H
        selected, leftover = self._select_rooms_by_area_greedy(self.rooms, cap)
        if not selected:
            return []

        layouts: list[Layout] = []
        sorters = [
            ("area_desc", "area"),
            ("along_desc", "along"),
            ("perp_desc", "perp"),
            ("area_asc", "area asc"),
        ]
        vx_pos = [
            self.W * 0.5 - self.CORRIDOR_WIDTH / 2,
            self.W / 3 - self.CORRIDOR_WIDTH / 2,
            self.W * 2 / 3 - self.CORRIDOR_WIDTH / 2,
        ]
        hy_pos = [
            self.H * 0.5 - self.CORRIDOR_WIDTH / 2,
            self.H / 3 - self.CORRIDOR_WIDTH / 2,
            self.H * 2 / 3 - self.CORRIDOR_WIDTH / 2,
        ]

        for key, _ in sorters:
            for cx in vx_pos:
                if lay := self._try_place_vertical(selected, cx, key):
                    lay.unplaced_names.extend([r.name for r in leftover])
                    layouts.append(lay)
            for cy in hy_pos:
                if lay := self._try_place_horizontal(selected, cy, key):
                    lay.unplaced_names.extend([r.name for r in leftover])
                    layouts.append(lay)

        seen, unique_layouts = set(), []
        for L in sorted(
            layouts,
            key=lambda l: (l.placed_count, sum(p.w * p.h for p in l.placed)),
            reverse=True,
        ):
            sig_val = round(
                (
                    L.corridor.x
                    if L.corridor.orientation == Orientation.VERTICAL
                    else L.corridor.y
                ),
                1,
            )
            sig = (
                L.corridor.orientation,
                sig_val,
                L.placed_count,
                tuple(sorted(p.name for p in L.placed)),
            )
            if sig not in seen:
                seen.add(sig)
                unique_layouts.append(L)
        return unique_layouts[:50]

    def _select_rooms_by_area_greedy(
        self, rooms: list[RoomSpec], max_area: float
    ) -> tuple[list[RoomSpec], list[RoomSpec]]:
        sorted_rooms = sorted(rooms, key=lambda r: r.w * r.h, reverse=True)
        selected, leftover, current_area = [], [], 0.0
        for room in sorted_rooms:
            if current_area + room.w * room.h <= max_area + EPS:
                selected.append(room)
                current_area += room.w * room.h
            else:
                leftover.append(room)
        return selected, leftover

    def _try_place_vertical(
        self, rooms: list[RoomSpec], cx: float, sort_key: str
    ) -> Layout | None:
        if not (0 <= cx and cx + self.CORRIDOR_WIDTH <= self.W):
            return None
        left_w, right_w = cx, self.W - (cx + self.CORRIDOR_WIDTH)
        if left_w <= 0 and right_w <= 0:
            return None

        ordered = self._sort_rooms(rooms, sort_key, is_vertical=True)
        y_l, y_r, placed, unplaced = 0.0, 0.0, [], []

        for r in ordered:
            placed_flag = False
            for side in (
                [Side.LEFT, Side.RIGHT] if y_l <= y_r else [Side.RIGHT, Side.LEFT]
            ):
                for rotated in [False, True]:
                    if not r.can_rotate and rotated:
                        continue
                    w, h = (r.h, r.w) if rotated else (r.w, r.h)

                    avail_w, curr_y = (
                        (left_w, y_l) if side == Side.LEFT else (right_w, y_r)
                    )
                    if w <= avail_w + EPS and curr_y + h <= self.H + EPS:
                        x0 = cx - w if side == Side.LEFT else cx + self.CORRIDOR_WIDTH
                        placed.append(PlacedRoom(r.name, x0, curr_y, w, h, rotated))
                        if side == Side.LEFT:
                            y_l += h
                        else:
                            y_r += h
                        placed_flag = True
                        break
                if placed_flag:
                    break
            if not placed_flag:
                unplaced.append(r.name)

        return (
            Layout(
                placed,
                unplaced,
                self.W,
                self.H,
                Corridor(cx, 0.0, self.CORRIDOR_WIDTH, self.H, Orientation.VERTICAL),
            )
            if placed
            else None
        )

    def _try_place_horizontal(
        self, rooms: list[RoomSpec], cy: float, sort_key: str
    ) -> Layout | None:
        if not (0 <= cy and cy + self.CORRIDOR_WIDTH <= self.H):
            return None
        top_h, bot_h = cy, self.H - (cy + self.CORRIDOR_WIDTH)
        if top_h <= 0 and bot_h <= 0:
            return None

        ordered = self._sort_rooms(rooms, sort_key, is_vertical=False)
        x_t, x_b, placed, unplaced = 0.0, 0.0, [], []

        for r in ordered:
            placed_flag = False
            for side in (
                [Side.TOP, Side.BOTTOM] if x_t <= x_b else [Side.BOTTOM, Side.TOP]
            ):
                for rotated in [False, True]:
                    if not r.can_rotate and rotated:
                        continue
                    w, h = (r.h, r.w) if rotated else (r.w, r.h)

                    avail_h, curr_x = (top_h, x_t) if side == Side.TOP else (bot_h, x_b)
                    if h <= avail_h + EPS and curr_x + w <= self.W + EPS:
                        y0 = cy - h if side == Side.TOP else cy + self.CORRIDOR_WIDTH
                        placed.append(PlacedRoom(r.name, curr_x, y0, w, h, rotated))
                        if side == Side.TOP:
                            x_t += w
                        else:
                            x_b += w
                        placed_flag = True
                        break
                if placed_flag:
                    break
            if not placed_flag:
                unplaced.append(r.name)

        return (
            Layout(
                placed,
                unplaced,
                self.W,
                self.H,
                Corridor(0.0, cy, self.W, self.CORRIDOR_WIDTH, Orientation.HORIZONTAL),
            )
            if placed
            else None
        )

    def _sort_rooms(
        self, rooms: list[RoomSpec], key: str, is_vertical: bool
    ) -> list[RoomSpec]:
        if key == "area_desc":
            return sorted(rooms, key=lambda r: r.w * r.h, reverse=True)
        if key == "area_asc":
            return sorted(rooms, key=lambda r: r.w * r.h)

        along = (lambda r: r.h) if is_vertical else (lambda r: r.w)
        perp = (lambda r: r.w) if is_vertical else (lambda r: r.h)

        if key == "along_desc":
            return sorted(
                rooms,
                key=lambda r: max(along(r), perp(r) if r.can_rotate else -1),
                reverse=True,
            )
        if key == "perp_desc":
            return sorted(
                rooms,
                key=lambda r: max(perp(r), along(r) if r.can_rotate else -1),
                reverse=True,
            )
        return rooms[:]
