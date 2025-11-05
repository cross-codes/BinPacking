import math
from constants import *
from typing import override
from algorithm.AbstractAlgorithm import AbstractAlgorithm
from ds.PlacedRoom import PlacedRoom
from ds.Layout import Layout
from ds.RoomSpec import RoomSpec


class Backtracker(AbstractAlgorithm):
    @override
    def generate(self) -> list[Layout]:
        layouts, seen_sigs = [], set()
        base_rooms = list(self.rooms)

        for _ in range(RANDOM_PERM_TRIES):
            if len(layouts) >= 50:
                break
            self.rng.shuffle(base_rooms)
            placed, unplaced = self._place_rooms_greedy(base_rooms)
            if not placed:
                continue

            layout = Layout(placed, [r.name for r in unplaced], self.W, self.H)
            self._find_and_set_entrance(layout)

            if self._is_accessible(layout):
                sig = (
                    tuple(
                        sorted(
                            (p.name, p.rotated, round(p.x, 1), round(p.y, 1))
                            for p in layout.placed
                        )
                    ),
                    tuple(sorted(layout.unplaced_names)),
                )
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    layouts.append(layout)
        return layouts

    def _place_rooms_greedy(
        self, order: list[RoomSpec]
    ) -> tuple[list[PlacedRoom], list[RoomSpec]]:
        placed, unplaced = [], []
        for r_spec in order:
            best_cand = None
            rotated = r_spec.can_rotate and self.rng.choice([True, False])
            w, h = (r_spec.h, r_spec.w) if rotated else (r_spec.w, r_spec.h)

            candidates = self._get_placement_candidates(w, h, placed)
            self.rng.shuffle(candidates)

            for x, y in candidates:
                cand_room = PlacedRoom(r_spec.name, x, y, w, h, rotated)
                if self._is_valid_move(cand_room, placed):
                    best_cand = cand_room
                    break

            if best_cand:
                placed.append(best_cand)
            else:
                unplaced.append(r_spec)
        return placed, unplaced

    def _get_placement_candidates(
        self, w: float, h: float, placed: list[PlacedRoom]
    ) -> list[tuple[float, float]]:
        test_xs, test_ys = {0.0, self.W - w}, {0.0, self.H - h}
        for pr in placed:
            test_xs.update({pr.x + pr.w, pr.x - w})
            test_ys.update({pr.y + pr.h, pr.y - h})
        candidates = set()
        for x in test_xs:
            candidates.update([(x, 0.0), (x, self.H - h)])
        for y in test_ys:
            candidates.update([(0.0, y), (self.W - w, y)])
        return list(candidates)

    def _is_valid_move(self, cand: PlacedRoom, placed: list[PlacedRoom]) -> bool:
        if not (
            cand.x >= -EPS
            and cand.y >= -EPS
            and cand.x + cand.w <= self.W + EPS
            and cand.y + cand.h <= self.H + EPS
        ):
            return False
        if not (
            abs(cand.y) <= EPS
            or abs(cand.x) <= EPS
            or abs(cand.x + cand.w - self.W) <= EPS
            or abs(cand.y + cand.h - self.H) <= EPS
        ):
            return False
        if any(self._rects_overlap(cand, p) for p in placed):
            return False

        temp_placed = placed + [cand]
        potential_entrances = self._get_potential_entrances(temp_placed)
        return bool(potential_entrances) and self._is_accessible_from_any(
            temp_placed, potential_entrances
        )

    def _is_accessible(self, layout: Layout) -> bool:
        return (
            self._is_accessible_from_any(layout.placed, [layout.entrance_pos])
            if layout.entrance_pos
            else False
        )

    def _is_accessible_from_any(
        self, placed: list[PlacedRoom], entrances: list[tuple[float, float]]
    ) -> bool:
        res = 1.5
        cols, rows = max(1, int(math.ceil(self.W / res))), max(
            1, int(math.ceil(self.H / res))
        )
        grid = [[True] * cols for _ in range(rows)]
        total_free_cells = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                cell_x, cell_y = c_idx * res + res / 2, r_idx * res + res / 2
                if any(
                    p.x - EPS <= cell_x < p.x + p.w + EPS
                    and p.y - EPS <= cell_y < p.y + p.h + EPS
                    for p in placed
                ):
                    grid[r_idx][c_idx] = False
                if grid[r_idx][c_idx]:
                    total_free_cells += 1

        if total_free_cells == 0:
            return True
        start_node = next(
            (
                (
                    min(cols - 1, max(0, int(ex / res))),
                    min(rows - 1, max(0, int(ey / res))),
                )
                for ex, ey in entrances
                if grid[min(rows - 1, max(0, int(ey / res)))][
                    min(cols - 1, max(0, int(ex / res)))
                ]
            ),
            None,
        )
        if start_node is None:
            return False

        q, visited, count = [start_node], {start_node}, 0
        while q:
            c, r = q.pop(0)
            count += 1
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nc, nr = c + dc, r + dr
                if (
                    0 <= nc < cols
                    and 0 <= nr < rows
                    and grid[nr][nc]
                    and (nc, nr) not in visited
                ):
                    visited.add((nc, nr))
                    q.append((nc, nr))
        return count == total_free_cells

    def _get_potential_entrances(
        self, placed: list[PlacedRoom]
    ) -> list[tuple[float, float]]:
        entrances, bounds = [], self._get_free_boundary_segments(placed)
        for edge, segs in bounds.items():
            for start, end in segs:
                mid = (start + end) / 2
                pos_map = {
                    "top": (mid, 0.0),
                    "bottom": (mid, self.H),
                    "left": (0.0, mid),
                    "right": (self.W, mid),
                }
                entrances.append(pos_map[edge])
        return entrances

    def _get_free_boundary_segments(self, placed: list[PlacedRoom]):
        def get_segs(total_len: float, occupied: list[tuple[float, float]]):
            occupied.sort()
            free, last_end = [], 0.0
            for start, end in occupied:
                if start > last_end + EPS:
                    free.append((last_end, start))
                last_end = max(last_end, end)
            if last_end < total_len - EPS:
                free.append((last_end, total_len))
            return free

        return {
            "top": get_segs(
                self.W, [(p.x, p.x + p.w) for p in placed if abs(p.y) < EPS]
            ),
            "bottom": get_segs(
                self.W,
                [(p.x, p.x + p.w) for p in placed if abs(p.y + p.h - self.H) < EPS],
            ),
            "left": get_segs(
                self.H, [(p.y, p.y + p.h) for p in placed if abs(p.x) < EPS]
            ),
            "right": get_segs(
                self.H,
                [(p.y, p.y + p.h) for p in placed if abs(p.x + p.w - self.W) < EPS],
            ),
        }

    def _find_and_set_entrance(self, layout: Layout):
        bounds = self._get_free_boundary_segments(layout.placed)
        layout.free_boundary_segments = bounds
        candidates = [
            (edge, seg)
            for edge, segs in bounds.items()
            for seg in segs
            if seg[1] - seg[0] > 1.0
        ]
        if not candidates:
            return
        edge, (start, end) = max(candidates, key=lambda item: item[1][1] - item[1][0])
        mid = (start + end) / 2
        pos_map = {
            "top": (mid, 0.0),
            "bottom": (mid, self.H),
            "left": (0.0, mid),
            "right": (self.W, mid),
        }
        layout.entrance_pos = pos_map.get(edge)

    def _rects_overlap(self, a: PlacedRoom, b: PlacedRoom) -> bool:
        return (
            a.x < b.x + b.w - EPS
            and a.x + a.w > b.x + EPS
            and a.y < b.y + b.h - EPS
            and a.y + a.h > b.y + EPS
        )
