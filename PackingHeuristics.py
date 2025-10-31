from struct.Layout import Layout
from struct.Corridor import Corridor, Orientation
from struct.PlacedRoom import PlacedRoom, Side
from struct.RoomSpec import RoomSpec
from constants import *


class PackingHeuristics:
    @staticmethod
    # Parse the room specification on the left
    def parse_rooms(text: str) -> list[RoomSpec]:
        rooms: list[RoomSpec] = []
        cnt: int = 1
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) == 2:
                try:
                    w = float(parts[0])
                    h = float(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid room line (expected 'w h'): {line}")
                rooms.append(RoomSpec(name=f"R{cnt}", width=w, height=h))
                cnt += 1

            elif len(parts) >= 3:
                name = parts[0]
                try:
                    w = float(parts[1])
                    h = float(parts[2])
                except ValueError:
                    raise ValueError(f"Invalid room line (expected 'name w h'): {line}")
                rooms.append(RoomSpec(name=name, width=w, height=h))

            else:
                raise ValueError(f"Invalid room line: {line}")
        return rooms

    @staticmethod
    # (1) Sort all rooms by area, descending
    # (2) Pick possible rooms until you run out of max area
    # Probably suboptimal?
    def select_rooms_by_area_greedy(
        rooms: list[RoomSpec], max_area: float
    ) -> tuple[list[RoomSpec], list[RoomSpec]]:

        sorted_area: list[RoomSpec] = sorted(
            rooms, key=lambda r: r.width * r.height, reverse=True
        )
        selected: list[RoomSpec] = []
        leftover: list[RoomSpec] = []
        acc: float = 0.0

        for room in sorted_area:
            area = room.width * room.height
            if acc + area <= max_area + 1e-9:
                selected.append(room)
                acc += area
            else:
                leftover.append(room)

        return selected, leftover

    @staticmethod
    # Rooms along a vertical corridor
    # (1) Place the vertical corridor at corridor_x
    # (2) Sort rooms using a strategy
    # (3) Iterate through sorted rooms to place place each room on some side
    # (4) Check for swapping height and width
    # Attempts to keep same height on either side
    def try_place_vertical(
        plot_w: float,
        plot_h: float,
        rooms: list[RoomSpec],
        corridor_x: float,
        sort_key: str,
        label_suffix: str,
    ) -> Layout | None:
        """
        Place rooms along a vertical corridor x=corridor_x..corridor_x+cw spanning height plot_h.
        Two stacks: left side (x decreasing from corridor), right side (x increasing from corridor).
        Stack along y from top (0) downward.
        """
        

        cw = CORRIDOR_WIDTH_UNITS
        if corridor_x < 0 or corridor_x + cw > plot_w:
            return None

        left_width = corridor_x
        right_width = plot_w - (corridor_x + cw)
        if left_width <= 0 and right_width <= 0:
            return None

        if sort_key == "area_desc":
            ordered = sorted(rooms, key=lambda r: r.width * r.height, reverse=True)
        elif sort_key == "along_desc":  # along corridor is height
            ordered = sorted(rooms, key=lambda r: max(r.height, r.width), reverse=True)
        elif (
            sort_key == "perp_desc"
        ):  # perpendicular to corridor is width used in side slab
            ordered = sorted(rooms, key=lambda r: max(r.width, r.height), reverse=True)
        elif sort_key == "area_asc":
            ordered = sorted(rooms, key=lambda r: r.width * r.height)
        else:
            ordered = rooms[:]

        y_left = 0.0
        y_right = 0.0
        placed: list[PlacedRoom] = []
        unplaced: list[RoomSpec] = []

        for r in ordered:
            candidates: list[tuple[Side, bool, float, float]] = []
            candidates.extend(
                [
                    (Side.LEFT, False, r.width, r.height),
                    (Side.RIGHT, False, r.width, r.height),
                    (Side.LEFT, True, r.height, r.width),
                    (Side.RIGHT, True, r.height, r.width),
                ]
            )

            side_order: list[Side] = (
                [Side.LEFT, Side.RIGHT]
                if y_left <= y_right
                else [Side.RIGHT, Side.LEFT]
            )

            placed_flag = False
            for side in side_order:
                for rotated in [False, True]:
                    if side == Side.LEFT:
                        avail_w = left_width
                        curr_y = y_left
                        x0 = corridor_x - (r.height if rotated else r.width)
                    else:
                        avail_w = right_width
                        curr_y = y_right
                        x0 = corridor_x + cw

                    room_w = r.height if rotated else r.width
                    room_h = r.width if rotated else r.height

                    if room_w <= avail_w + 1e-9 and curr_y + room_h <= plot_h + 1e-9:
                        y0 = curr_y
                        placed.append(
                            PlacedRoom(
                                name=r.name,
                                width=room_w,
                                height=room_h,
                                x=x0,
                                y=y0,
                                rotated=rotated,
                                side=side,
                            )
                        )
                        if side == Side.LEFT:
                            y_left += room_h
                        else:
                            y_right += room_h
                        placed_flag = True
                        break
                if placed_flag:
                    break

            if not placed_flag:
                unplaced.append(r)

        rooms_area = sum(pr.width * pr.height for pr in placed)
        layout = Layout(
            placed=placed,
            corridor=Corridor(
                x=corridor_x,
                y=0.0,
                width=cw,
                height=plot_h,
                orientation=Orientation.VERTICAL,
            ),
            plot_w=plot_w,
            plot_h=plot_h,
            rooms_area=rooms_area,
            placed_count=len(placed),
            label=f"Vertical @x={corridor_x:.2f} ({label_suffix})",
            unplaced=unplaced,
        )
        # tag for UI filtering
        setattr(layout, "heuristic", "single-vertical")
        setattr(layout, "variant", sort_key)  # optional: which sort key
        return layout

    @staticmethod
    # Same as above, but horizontal corridor
    def try_place_horizontal(
        plot_w: float,
        plot_h: float,
        rooms: list[RoomSpec],
        corridor_y: float,
        sort_key: str,
        label_suffix: str,
    ) -> Layout | None:

        ch = CORRIDOR_WIDTH_UNITS
        if corridor_y < 0 or corridor_y + ch > plot_h:
            return None

        top_height = corridor_y
        bottom_height = plot_h - (corridor_y + ch)
        if top_height <= 0 and bottom_height <= 0:
            return None

        if sort_key == "area_desc":
            ordered = sorted(rooms, key=lambda r: r.width * r.height, reverse=True)
        elif sort_key == "along_desc":  # along corridor is width now
            ordered = sorted(rooms, key=lambda r: max(r.width, r.height), reverse=True)
        elif sort_key == "perp_desc":  # perpendicular to corridor is height
            ordered = sorted(rooms, key=lambda r: max(r.width, r.height), reverse=True)
        elif sort_key == "area_asc":
            ordered = sorted(rooms, key=lambda r: r.width * r.height)
        else:
            ordered = rooms[:]

        x_top = 0.0
        x_bottom = 0.0
        placed: list[PlacedRoom] = []
        unplaced: list[RoomSpec] = []

        for r in ordered:
            side_order = (
                [Side.TOP, Side.BOTTOM]
                if x_top <= x_bottom
                else [Side.BOTTOM, Side.TOP]
            )
            placed_flag = False
            for side in side_order:
                for rotated in [False, True]:
                    if side == Side.TOP:
                        avail_h = top_height
                        curr_x = x_top
                        y0 = corridor_y - (r.height if rotated else r.width)
                    else:
                        avail_h = bottom_height
                        curr_x = x_bottom
                        y0 = corridor_y + ch

                    room_h = r.height if rotated else r.width
                    room_w = r.width if rotated else r.height

                    if room_h <= avail_h + 1e-9 and curr_x + room_w <= plot_w + 1e-9:
                        x0 = curr_x
                        placed.append(
                            PlacedRoom(
                                name=r.name,
                                width=room_w,
                                height=room_h,
                                x=x0,
                                y=y0,
                                rotated=rotated,
                                side=side,
                            )
                        )
                        if side == Side.TOP:
                            x_top += room_w
                        else:
                            x_bottom += room_w
                        placed_flag = True
                        break
                if placed_flag:
                    break
            if not placed_flag:
                unplaced.append(r)

        rooms_area = sum(pr.width * pr.height for pr in placed)
        layout = Layout(
            placed=placed,
            corridor=Corridor(
                x=0.0,
                y=corridor_y,
                width=plot_w,
                height=ch,
                orientation=Orientation.HORIZONTAL,
            ),
            plot_w=plot_w,
            plot_h=plot_h,
            rooms_area=rooms_area,
            placed_count=len(placed),
            label=f"Horizontal @y={corridor_y:.2f} ({label_suffix})",
            unplaced=unplaced,
        )
        # tag for UI filtering
        setattr(layout, "heuristic", "single-horizontal")
        setattr(layout, "variant", sort_key)  # optional
        return layout

    @staticmethod
    def generate_layouts(
        plot_w: float,
        plot_h: float,
        rooms: list[RoomSpec],
        desired_k: int | None = None,   # pass None to auto-pick max feasible k
    ) -> list[Layout]:
        if plot_w <= 0 or plot_h <= 0:
            return []

        # Select rooms under the global area cap first (70% if AREA_FRACTION_LIMIT=0.7)
        cap = AREA_FRACTION_LIMIT * plot_w * plot_h
        selected, leftover = PackingHeuristics.select_rooms_by_area_greedy(rooms, cap)
        if not selected:
            return []

        layouts: list[Layout] = []

        # ---- Try multi-corridor vertical layouts first (optional exact k via desired_k) ----
        multi = PackingHeuristics.try_place_multiple_vertical_corridors(
            plot_w=plot_w,
            plot_h=plot_h,
            rooms=selected,
            cw=CORRIDOR_WIDTH_UNITS,
            target_area_fraction=AREA_FRACTION_LIMIT,
            desired_k=desired_k,
        )
        if multi is not None and getattr(multi, "placed", None):
            multi.unplaced = leftover + getattr(multi, "unplaced", [])
            layouts.append(multi)

        # ---- Existing single-corridor heuristics ----
        sorters = [
            ("area_desc", "rooms by area desc"),
            ("along_desc", "by along-corridor size desc"),
            ("perp_desc", "by perpendicular size desc"),
            ("area_asc", "rooms by area asc"),
        ]

        vx_positions = [plot_w * 0.5, plot_w * (1 / 3), plot_w * (2 / 3)]
        hy_positions = [plot_h * 0.5, plot_h * (1 / 3), plot_h * (2 / 3)]

        for key, desc in sorters:
            for cx in vx_positions:
                lay = PackingHeuristics.try_place_vertical(plot_w, plot_h, selected, cx, key, desc)
                if lay is not None and lay.placed:
                    lay.unplaced = leftover + lay.unplaced
                    layouts.append(lay)

            for cy in hy_positions:
                lay = PackingHeuristics.try_place_horizontal(plot_w, plot_h, selected, cy, key, desc)
                if lay is not None and lay.placed:
                    lay.unplaced = leftover + lay.unplaced
                    layouts.append(lay)

        # ---- De-duplicate and sort ----
        layouts.sort(key=lambda L: (L.placed_count, L.rooms_area), reverse=True)
        seen: set[tuple[Orientation, float, int]] = set()
        unique: list[Layout] = []
        for L in layouts:
            if L.corridor.orientation == Orientation.VERTICAL:
                sig = (L.corridor.orientation, round(L.corridor.x, 2), L.placed_count)
            else:
                sig = (L.corridor.orientation, round(L.corridor.y, 2), L.placed_count)
            if sig not in seen:
                seen.add(sig)
                unique.append(L)

        return unique



    @staticmethod
    def try_place_multiple_vertical_corridors(
        plot_w: float,
        plot_h: float,
        rooms: list[RoomSpec],
        cw: float = CORRIDOR_WIDTH_UNITS,
        target_area_fraction: float = 0.7,
        desired_k: int | None = None,   # None = auto (try max feasible). Otherwise try this exact k.
    ) -> Layout | None:
        """
        Place as many (or exactly desired_k) parallel vertical corridors (width=cw) as possible,
        while keeping packed room area <= target_area_fraction * plot area.
        Corridors split the plot into (k+1) vertical slabs; rooms are packed per slab top-down, allowing rotation.
        Returns the first feasible layout found (max k first if desired_k is None).
        """
        if plot_w <= 0 or plot_h <= 0:
            return None

        plot_area = plot_w * plot_h
        room_area_budget = target_area_fraction * plot_area
        if room_area_budget <= 0:
            return None

        # shortest side of any room (used to quickly rule out too-narrow slabs)
        if not rooms:
            return None
        min_short_side = min(min(r.width, r.height) for r in rooms)

        # k search order
        max_possible_k = int(plot_w // cw)  # loose bound
        if max_possible_k < 1:
            return None

        if desired_k is not None:
            k0 = max(1, min(desired_k, max_possible_k))
            k_values = [k0]  # exact-k behavior
        else:
            k_values = list(range(max_possible_k, 0, -1))  # try most corridors first

        for k in k_values:
            total_corridor_width = k * cw
            if total_corridor_width >= plot_w - 1e-9:
                continue

            total_slabs_width = plot_w - total_corridor_width
            num_slabs = k + 1
            slab_width = total_slabs_width / num_slabs

            # slab must be wide enough to fit at least the narrowest room side
            if slab_width + 1e-9 < min_short_side:
                continue

            # area left for rooms after corridors
            available_area_for_rooms = plot_area - (k * cw * plot_h)
            if available_area_for_rooms + 1e-9 < room_area_budget:
                # carving too many corridors leaves insufficient area to reach the 70% budget
                continue

            # define slabs (x positions)
            slabs: list[dict] = []
            x_cursor = 0.0
            for i in range(num_slabs):
                slabs.append({"x": x_cursor, "width": slab_width, "used_height": 0.0})
                x_cursor += slab_width
                if i < k:
                    x_cursor += cw  # skip a corridor

            # greedy pack into slabs (top-down), larger rooms first
            placed: list[PlacedRoom] = []
            unplaced: list[RoomSpec] = []
            ordered = sorted(rooms, key=lambda r: r.width * r.height, reverse=True)

            for r in ordered:
                placed_flag = False
                for slab in sorted(slabs, key=lambda s: s["used_height"]):
                    for rotated in (False, True):
                        room_w = r.height if rotated else r.width
                        room_h = r.width  if rotated else r.height
                        if room_w <= slab["width"] + 1e-9 and slab["used_height"] + room_h <= plot_h + 1e-9:
                            x0 = slab["x"]
                            y0 = slab["used_height"]
                            placed.append(
                                PlacedRoom(
                                    name=r.name,
                                    width=room_w,
                                    height=room_h,
                                    x=x0,
                                    y=y0,
                                    rotated=rotated,
                                    side=None,  # slab index not tracked
                                )
                            )
                            slab["used_height"] += room_h
                            placed_flag = True
                            break
                    if placed_flag:
                        break
                if not placed_flag:
                    unplaced.append(r)

            rooms_area = sum(pr.width * pr.height for pr in placed)
            if rooms_area <= room_area_budget + 1e-9:
                # construct corridor objects (between slabs)
                corridors: list[Corridor] = []
                x_cursor = slab_width
                for _ in range(k):
                    corridors.append(
                        Corridor(
                            x=x_cursor,
                            y=0.0,
                            width=cw,
                            height=plot_h,
                            orientation=Orientation.VERTICAL,
                        )
                    )
                    x_cursor += cw + slab_width

                label = f"{k} vertical corridors (cw={cw})"
                layout = Layout(
                    placed=placed,
                    corridor=corridors[0] if corridors else None,  # legacy single-corridor field
                    plot_w=plot_w,
                    plot_h=plot_h,
                    rooms_area=rooms_area,
                    placed_count=len(placed),
                    label=label,
                    unplaced=unplaced,
                )
                # attach extras for the UI
                setattr(layout, "corridors", corridors)
                setattr(layout, "heuristic", "multi-vertical")
                setattr(layout, "k_corridors", len(corridors))
                return layout

        # nothing feasible
        return None
