#!/usr/bin/env python3
"""
Final, single-file version of the multi-algorithm layout generator.
Supports choosing between "Boundary Packer" and "Single Corridor" algorithms.
Includes partial placements, connectivity checks, and dynamic entrances.
"""
import math
import random
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Tuple, Optional, Dict, Type
from enum import Enum

# --- Constants and Tunables ---
RANDOM_PERM_TRIES = 500
SEED = 24680
EPS = 1e-7

# --- Universal Data Structures ---
class RoomSpec:
    def __init__(self, name: str, w: float, h: float, can_rotate: bool = True):
        self.name, self.w, self.h, self.can_rotate = name, float(w), float(h), can_rotate

class PlacedRoom:
    def __init__(self, name: str, x: float, y: float, w: float, h: float, rotated: bool):
        self.name, self.x, self.y, self.w, self.h, self.rotated = name, float(x), float(y), float(w), float(h), bool(rotated)

class Orientation(Enum):
    VERTICAL = 1
    HORIZONTAL = 2

class Side(Enum):
    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 3, 4

class Corridor:
    def __init__(self, x: float, y: float, w: float, h: float, orientation: Orientation):
        self.x, self.y, self.w, self.h, self.orientation = x, y, w, h, orientation

class Layout:
    def __init__(self, placed: List[PlacedRoom], unplaced: List[str], plot_w: float, plot_h: float, corridor: Optional[Corridor] = None):
        self.placed, self.unplaced_names, self.plot_w, self.plot_h = placed, unplaced, plot_w, plot_h
        self.corridor = corridor
        self.placed_count = len(placed)
        self.entrance_pos: Optional[Tuple[float, float]] = None
        self.free_boundary_segments: Optional[Dict[str, List[Tuple[float, float]]]] = None

# --- Abstract Base Class for Algorithms ---
class LayoutAlgorithm:
    def __init__(self, plot_w: float, plot_h: float, rooms: List[RoomSpec]):
        self.W, self.H, self.rooms = plot_w, plot_h, rooms
        self.rng = random.Random(SEED)

    def generate(self) -> List[Layout]:
        raise NotImplementedError

# --- Algorithm 1: Boundary Packer ---
class BoundaryPackerAlgorithm(LayoutAlgorithm):
    def generate(self) -> List[Layout]:
        layouts, seen_sigs = [], set()
        base_rooms = list(self.rooms)
        
        for _ in range(RANDOM_PERM_TRIES):
            if len(layouts) >= 50: break
            self.rng.shuffle(base_rooms)
            placed, unplaced = self._place_rooms_greedy(base_rooms)
            if not placed: continue

            layout = Layout(placed, [r.name for r in unplaced], self.W, self.H)
            self._find_and_set_entrance(layout)
            
            if self._is_accessible(layout):
                sig = (tuple(sorted((p.name, p.rotated, round(p.x,1), round(p.y,1)) for p in layout.placed)), tuple(sorted(layout.unplaced_names)))
                if sig not in seen_sigs:
                    seen_sigs.add(sig); layouts.append(layout)
        return layouts

    def _place_rooms_greedy(self, order: List[RoomSpec]) -> Tuple[List[PlacedRoom], List[RoomSpec]]:
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
                    best_cand = cand_room; break
            
            if best_cand: placed.append(best_cand)
            else: unplaced.append(r_spec)
        return placed, unplaced

    def _get_placement_candidates(self, w: float, h: float, placed: List[PlacedRoom]) -> List[Tuple[float, float]]:
        test_xs, test_ys = {0.0, self.W - w}, {0.0, self.H - h}
        for pr in placed:
            test_xs.update({pr.x + pr.w, pr.x - w})
            test_ys.update({pr.y + pr.h, pr.y - h})
        candidates = set()
        for x in test_xs: candidates.update([(x, 0.0), (x, self.H - h)])
        for y in test_ys: candidates.update([(0.0, y), (self.W - w, y)])
        return list(candidates)

    def _is_valid_move(self, cand: PlacedRoom, placed: List[PlacedRoom]) -> bool:
        if not (cand.x >= -EPS and cand.y >= -EPS and cand.x + cand.w <= self.W + EPS and cand.y + cand.h <= self.H + EPS): return False
        if not (abs(cand.y)<=EPS or abs(cand.x)<=EPS or abs(cand.x+cand.w-self.W)<=EPS or abs(cand.y+cand.h-self.H)<=EPS): return False
        if any(self._rects_overlap(cand, p) for p in placed): return False
        
        temp_placed = placed + [cand]
        potential_entrances = self._get_potential_entrances(temp_placed)
        return bool(potential_entrances) and self._is_accessible_from_any(temp_placed, potential_entrances)

    def _is_accessible(self, layout: Layout) -> bool:
        return self._is_accessible_from_any(layout.placed, [layout.entrance_pos]) if layout.entrance_pos else False

    def _is_accessible_from_any(self, placed: List[PlacedRoom], entrances: List[Tuple[float,float]]) -> bool:
        res = 1.5 # Grid resolution for connectivity check
        cols, rows = max(1, int(math.ceil(self.W / res))), max(1, int(math.ceil(self.H / res)))
        grid = [[True] * cols for _ in range(rows)]
        total_free_cells = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                cell_x, cell_y = c_idx * res + res/2, r_idx * res + res/2
                if any(p.x - EPS <= cell_x < p.x + p.w + EPS and p.y - EPS <= cell_y < p.y + p.h + EPS for p in placed):
                    grid[r_idx][c_idx] = False
                if grid[r_idx][c_idx]: total_free_cells += 1
        
        if total_free_cells == 0: return True
        start_node = next(( (min(cols-1, max(0, int(ex/res))), min(rows-1, max(0, int(ey/res)))) for ex, ey in entrances if grid[min(rows-1, max(0, int(ey/res)))][min(cols-1, max(0, int(ex/res)))] ), None)
        if start_node is None: return False

        q, visited, count = [start_node], {start_node}, 0
        while q:
            c, r = q.pop(0); count += 1
            for dc, dr in [(0,1), (0,-1), (1,0), (-1,0)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < cols and 0 <= nr < rows and grid[nr][nc] and (nc, nr) not in visited:
                    visited.add((nc, nr)); q.append((nc, nr))
        return count == total_free_cells

    def _get_potential_entrances(self, placed: List[PlacedRoom]) -> List[Tuple[float,float]]:
        entrances, bounds = [], self._get_free_boundary_segments(placed)
        for edge, segs in bounds.items():
            for start, end in segs:
                mid = (start + end) / 2
                pos_map = {'top': (mid, 0.0), 'bottom': (mid, self.H), 'left': (0.0, mid), 'right': (self.W, mid)}
                entrances.append(pos_map[edge])
        return entrances

    def _get_free_boundary_segments(self, placed: List[PlacedRoom]):
        def get_segs(total_len: float, occupied: List[Tuple[float, float]]):
            occupied.sort()
            free, last_end = [], 0.0
            for start, end in occupied:
                if start > last_end + EPS: free.append((last_end, start))
                last_end = max(last_end, end)
            if last_end < total_len - EPS: free.append((last_end, total_len))
            return free
        return {'top': get_segs(self.W, [(p.x, p.x + p.w) for p in placed if abs(p.y) < EPS]),
                'bottom': get_segs(self.W, [(p.x, p.x + p.w) for p in placed if abs(p.y + p.h - self.H) < EPS]),
                'left': get_segs(self.H, [(p.y, p.y + p.h) for p in placed if abs(p.x) < EPS]),
                'right': get_segs(self.H, [(p.y, p.y + p.h) for p in placed if abs(p.x + p.w - self.W) < EPS])}

    def _find_and_set_entrance(self, layout: Layout):
        bounds = self._get_free_boundary_segments(layout.placed)
        layout.free_boundary_segments = bounds # Store segments for drawing
        candidates = [(edge, seg) for edge, segs in bounds.items() for seg in segs if seg[1] - seg[0] > 1.0]
        if not candidates: return
        edge, (start, end) = max(candidates, key=lambda item: item[1][1] - item[1][0])
        mid = (start + end) / 2
        pos_map = {'top': (mid, 0.0), 'bottom': (mid, self.H), 'left': (0.0, mid), 'right': (self.W, mid)}
        layout.entrance_pos = pos_map.get(edge)

    def _rects_overlap(self, a: PlacedRoom, b: PlacedRoom) -> bool:
        return a.x < b.x + b.w - EPS and a.x + a.w > b.x + EPS and a.y < b.y + b.h - EPS and a.y + a.h > b.y + EPS

# --- Algorithm 2: Single Corridor ---
class SingleCorridorAlgorithm(LayoutAlgorithm):
    def __init__(self, plot_w: float, plot_h: float, rooms: List[RoomSpec]):
        super().__init__(plot_w, plot_h, rooms)
        self.CORRIDOR_WIDTH = 2.0
        self.AREA_FRACTION_LIMIT = 0.9

    def generate(self) -> List[Layout]:
        if self.W <= 0 or self.H <= 0: return []
        
        cap = self.AREA_FRACTION_LIMIT * self.W * self.H
        selected, leftover = self._select_rooms_by_area_greedy(self.rooms, cap)
        if not selected: return []

        layouts: List[Layout] = []
        sorters = [("area_desc", "area"), ("along_desc", "along"), ("perp_desc", "perp"), ("area_asc", "area asc")]
        vx_pos = [self.W*0.5 - self.CORRIDOR_WIDTH/2, self.W/3 - self.CORRIDOR_WIDTH/2, self.W*2/3 - self.CORRIDOR_WIDTH/2]
        hy_pos = [self.H*0.5 - self.CORRIDOR_WIDTH/2, self.H/3 - self.CORRIDOR_WIDTH/2, self.H*2/3 - self.CORRIDOR_WIDTH/2]

        for key, _ in sorters:
            for cx in vx_pos:
                if lay := self._try_place_vertical(selected, cx, key):
                    lay.unplaced_names.extend([r.name for r in leftover]); layouts.append(lay)
            for cy in hy_pos:
                if lay := self._try_place_horizontal(selected, cy, key):
                    lay.unplaced_names.extend([r.name for r in leftover]); layouts.append(lay)
        
        seen, unique_layouts = set(), []
        for L in sorted(layouts, key=lambda l: (l.placed_count, sum(p.w*p.h for p in l.placed)), reverse=True):
            sig_val = round(L.corridor.x if L.corridor.orientation == Orientation.VERTICAL else L.corridor.y, 1)
            sig = (L.corridor.orientation, sig_val, L.placed_count, tuple(sorted(p.name for p in L.placed)))
            if sig not in seen:
                seen.add(sig); unique_layouts.append(L)
        return unique_layouts[:50]

    def _select_rooms_by_area_greedy(self, rooms: List[RoomSpec], max_area: float) -> Tuple[List[RoomSpec], List[RoomSpec]]:
        sorted_rooms = sorted(rooms, key=lambda r: r.w * r.h, reverse=True)
        selected, leftover, current_area = [], [], 0.0
        for room in sorted_rooms:
            if current_area + room.w * room.h <= max_area + EPS:
                selected.append(room); current_area += room.w * room.h
            else:
                leftover.append(room)
        return selected, leftover

    def _try_place_vertical(self, rooms: List[RoomSpec], cx: float, sort_key: str) -> Optional[Layout]:
        if not (0 <= cx and cx + self.CORRIDOR_WIDTH <= self.W): return None
        left_w, right_w = cx, self.W - (cx + self.CORRIDOR_WIDTH)
        if left_w <= 0 and right_w <= 0: return None

        ordered = self._sort_rooms(rooms, sort_key, is_vertical=True)
        y_l, y_r, placed, unplaced = 0.0, 0.0, [], []

        for r in ordered:
            placed_flag = False
            for side in ([Side.LEFT, Side.RIGHT] if y_l <= y_r else [Side.RIGHT, Side.LEFT]):
                for rotated in [False, True]:
                    if not r.can_rotate and rotated: continue
                    w, h = (r.h, r.w) if rotated else (r.w, r.h)
                    
                    avail_w, curr_y = (left_w, y_l) if side == Side.LEFT else (right_w, y_r)
                    if w <= avail_w + EPS and curr_y + h <= self.H + EPS:
                        x0 = cx - w if side == Side.LEFT else cx + self.CORRIDOR_WIDTH
                        placed.append(PlacedRoom(r.name, x0, curr_y, w, h, rotated))
                        if side == Side.LEFT: y_l += h
                        else: y_r += h
                        placed_flag = True; break
                if placed_flag: break
            if not placed_flag: unplaced.append(r.name)

        return Layout(placed, unplaced, self.W, self.H, Corridor(cx, 0.0, self.CORRIDOR_WIDTH, self.H, Orientation.VERTICAL)) if placed else None

    def _try_place_horizontal(self, rooms: List[RoomSpec], cy: float, sort_key: str) -> Optional[Layout]:
        if not (0 <= cy and cy + self.CORRIDOR_WIDTH <= self.H): return None
        top_h, bot_h = cy, self.H - (cy + self.CORRIDOR_WIDTH)
        if top_h <= 0 and bot_h <= 0: return None

        ordered = self._sort_rooms(rooms, sort_key, is_vertical=False)
        x_t, x_b, placed, unplaced = 0.0, 0.0, [], []
        
        for r in ordered:
            placed_flag = False
            for side in ([Side.TOP, Side.BOTTOM] if x_t <= x_b else [Side.BOTTOM, Side.TOP]):
                for rotated in [False, True]:
                    if not r.can_rotate and rotated: continue
                    w, h = (r.h, r.w) if rotated else (r.w, r.h)
                    
                    avail_h, curr_x = (top_h, x_t) if side == Side.TOP else (bot_h, x_b)
                    if h <= avail_h + EPS and curr_x + w <= self.W + EPS:
                        y0 = cy - h if side == Side.TOP else cy + self.CORRIDOR_WIDTH
                        placed.append(PlacedRoom(r.name, curr_x, y0, w, h, rotated))
                        if side == Side.TOP: x_t += w
                        else: x_b += w
                        placed_flag = True; break
                if placed_flag: break
            if not placed_flag: unplaced.append(r.name)

        return Layout(placed, unplaced, self.W, self.H, Corridor(0.0, cy, self.W, self.CORRIDOR_WIDTH, Orientation.HORIZONTAL)) if placed else None

    def _sort_rooms(self, rooms: List[RoomSpec], key: str, is_vertical: bool) -> List[RoomSpec]:
        if key == "area_desc": return sorted(rooms, key=lambda r: r.w * r.h, reverse=True)
        if key == "area_asc": return sorted(rooms, key=lambda r: r.w * r.h)
        
        along = (lambda r: r.h) if is_vertical else (lambda r: r.w)
        perp = (lambda r: r.w) if is_vertical else (lambda r: r.h)
        
        if key == "along_desc": return sorted(rooms, key=lambda r: max(along(r), perp(r) if r.can_rotate else -1), reverse=True)
        if key == "perp_desc": return sorted(rooms, key=lambda r: max(perp(r), along(r) if r.can_rotate else -1), reverse=True)
        return rooms[:]

# --- Main Application Class ---
ALGORITHMS: Dict[str, Type[LayoutAlgorithm]] = {
    "Boundary Packer": BoundaryPackerAlgorithm,
    "Single Corridor": SingleCorridorAlgorithm,
}

class App:
    def __init__(self, root):
        self.root = root
        root.title("Multi-Algorithm Layout Generator")
        
        top = ttk.Frame(root); top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        main = ttk.Frame(root); main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        bottom = ttk.Frame(root); bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        
        ttk.Label(top, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value=list(ALGORITHMS.keys())[0])

        algo_dropdown = ttk.Combobox(top, textvariable=self.algo_var, values=list(ALGORITHMS.keys()), state="readonly")
        algo_dropdown.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(top, text="Plot W:").pack(side=tk.LEFT)
        self.w_var = tk.DoubleVar(value=20.0)
        ttk.Entry(top, width=7, textvariable=self.w_var).pack(side=tk.LEFT, padx=(0,4))
        ttk.Label(top, text="H:").pack(side=tk.LEFT)
        self.h_var = tk.DoubleVar(value=25.0)
        ttk.Entry(top, width=7, textvariable=self.h_var).pack(side=tk.LEFT, padx=(0,10))
        
        ttk.Button(top, text="Generate", command=self.generate).pack(side=tk.RIGHT, padx=6)
        
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.Y, padx=(6,0), pady=6)
        ttk.Label(left, text="Rooms (name w h [rotatable:1/0]):").pack(anchor=tk.W)
        self.text = ScrolledText(left, width=30, height=28); self.text.pack(fill=tk.Y)
        self.text.insert("1.0", "R1 6 5 1\nR2 7 4 1\nR3 4 4 0\nR4 5 5 0\nR5 3 8 1\nR6 6 6 0\nR7 3 9 1\nR8 4 7 1\nR9 3 3 0\nR10 4.5 4 1")
        self.canvas = tk.Canvas(main, bg="white"); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Button(bottom, text="<< Prev", command=self.prev).pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom, text="Next >>", command=self.next).pack(side=tk.LEFT)
        self.info = ttk.Label(bottom, text="No layouts"); self.info.pack(side=tk.LEFT, padx=10)
        self.status = ttk.Label(bottom, text="Ready"); self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        
        self.layouts, self.idx, self.total_rooms = [], 0, 0
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.generate()

    def generate(self):
        self.status.config(text="Generating...", foreground="black"); self.root.update_idletasks()
        rooms = self.parse_rooms(); self.total_rooms = len(rooms)
        if not rooms: self.status.config(text="No rooms.", foreground="red"); return

        AlgorithmClass = ALGORITHMS[self.algo_var.get()]
        gen = AlgorithmClass(self.w_var.get(), self.h_var.get(), rooms)
        self.layouts = gen.generate(); self.idx = 0
        
        self.status.config(text=f"Found {len(self.layouts)} valid layouts."); self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts:
            self.info.config(text="No valid layouts."); self.status.config(text="Try a larger plot or different algorithm.", foreground="red"); return

        layout = self.layouts[self.idx]
        pad = 60 # Increased padding to make space for total dimension lines
        W, H = layout.plot_w, layout.plot_h
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        scale = min((cw - 2*pad) / W, (ch - 2*pad) / H) if W*H>0 else 1
        ox, oy = (cw - W*scale)/2, (ch - H*scale)/2

        if layout.corridor:
            c = layout.corridor
            self.canvas.create_rectangle(ox + c.x*scale, oy + c.y*scale, ox + (c.x+c.w)*scale, oy + (c.y+c.h)*scale, fill="#d0d0d0", outline="")
        else:
             self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, fill="#d0d0d0", outline="")

        for r in layout.placed:
            x1, y1, x2, y2 = ox+r.x*scale, oy+r.y*scale, ox+(r.x+r.w)*scale, oy+(r.y+r.h)*scale
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4ea3ff", outline="#003366", width=1.5)
            name_tag = f"{r.name}*" if r.rotated else r.name
            self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f"{name_tag}\n{r.w:.1f}x{r.h:.1f}", fill="white", font=("Arial",9))
        
        self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, width=2, outline="#333")

        # For Boundary Packer, draw dimension lines
        if not layout.corridor:
            self._draw_dimension_lines(layout, scale, ox, oy)
            
        if layout.entrance_pos:
            ex, ey = ox + layout.entrance_pos[0]*scale, oy + layout.entrance_pos[1]*scale
            is_vertical_edge = abs(layout.entrance_pos[0]) < EPS or abs(layout.entrance_pos[0] - W) < EPS
            dx, dy = (0, 10) if is_vertical_edge else (10, 0)
            self.canvas.create_line(ex-dx, ey-dy, ex+dx, ey+dy, fill="red", width=5)

        self.info.config(text=f"Layout {self.idx+1}/{len(self.layouts)} — Placed {layout.placed_count}/{self.total_rooms}")
        status_text = "All rooms placed." if not layout.unplaced_names else f"Unplaced: {', '.join(layout.unplaced_names)}"
        self.status.config(text=status_text, foreground="green" if not layout.unplaced_names else "red")
        
    def _draw_dimension_lines(self, layout, scale, ox, oy):
        W, H = layout.plot_w, layout.plot_h
        font_size = 8
        offset = 15 
        tick_size = 3
        min_len_to_show = 0.5 
        line_color = "#555555"
        text_padding = 3

        # Mark Origin (0,0)
        self.canvas.create_line(ox-5, oy-5, ox+5, oy+5, fill="red", width=1.5)
        self.canvas.create_line(ox-5, oy+5, ox+5, oy-5, fill="red", width=1.5)
        self.canvas.create_text(ox - 8, oy - 8, text="(0,0)", font=("Arial", font_size), fill="red", anchor=tk.SE)

        segs = layout.free_boundary_segments
        if segs:
            for start, end in segs.get('top', []):
                length = end - start
                if length > min_len_to_show:
                    x1, x2 = ox + start * scale, ox + end * scale
                    y_pos = oy - offset
                    self.canvas.create_line(x1, y_pos - tick_size, x1, y_pos + tick_size, fill=line_color)
                    self.canvas.create_line(x2, y_pos - tick_size, x2, y_pos + tick_size, fill=line_color)
                    self.canvas.create_line(x1, y_pos, x2, y_pos, fill=line_color)
                    self.canvas.create_text((x1+x2)/2, y_pos - text_padding, text=f"{length:.1f}", font=("Arial", font_size), fill=line_color, anchor=tk.S)

            for start, end in segs.get('bottom', []):
                length = end - start
                if length > min_len_to_show:
                    x1, x2 = ox + start * scale, ox + end * scale
                    y_pos = oy + H * scale + offset
                    self.canvas.create_line(x1, y_pos - tick_size, x1, y_pos + tick_size, fill=line_color)
                    self.canvas.create_line(x2, y_pos - tick_size, x2, y_pos + tick_size, fill=line_color)
                    self.canvas.create_line(x1, y_pos, x2, y_pos, fill=line_color)
                    self.canvas.create_text((x1+x2)/2, y_pos + text_padding, text=f"{length:.1f}", font=("Arial", font_size), fill=line_color, anchor=tk.N)

            for start, end in segs.get('left', []):
                length = end - start
                if length > min_len_to_show:
                    y1, y2 = oy + start * scale, oy + end * scale
                    x_pos = ox - offset
                    self.canvas.create_line(x_pos - tick_size, y1, x_pos + tick_size, y1, fill=line_color)
                    self.canvas.create_line(x_pos - tick_size, y2, x_pos + tick_size, y2, fill=line_color)
                    self.canvas.create_line(x_pos, y1, x_pos, y2, fill=line_color)
                    self.canvas.create_text(x_pos - text_padding, (y1+y2)/2, text=f"{length:.1f}", font=("Arial", font_size), fill=line_color, anchor=tk.E)

            for start, end in segs.get('right', []):
                length = end - start
                if length > min_len_to_show:
                    y1, y2 = oy + start * scale, oy + end * scale
                    x_pos = ox + W * scale + offset
                    self.canvas.create_line(x_pos - tick_size, y1, x_pos + tick_size, y1, fill=line_color)
                    self.canvas.create_line(x_pos - tick_size, y2, x_pos + tick_size, y2, fill=line_color)
                    self.canvas.create_line(x_pos, y1, x_pos, y2, fill=line_color)
                    self.canvas.create_text(x_pos + text_padding, (y1+y2)/2, text=f"{length:.1f}", font=("Arial", font_size), fill=line_color, anchor=tk.W)

        # Draw total W and H dimension lines
        total_dim_offset = offset + 27 # Increased this value for more separation
        total_line_color = "#00008B" # Dark Blue

        # Total Width (W)
        x1, x2 = ox, ox + W * scale
        y_pos = oy - total_dim_offset
        self.canvas.create_line(x1, y_pos - tick_size, x1, y_pos + tick_size, fill=total_line_color)
        self.canvas.create_line(x2, y_pos - tick_size, x2, y_pos + tick_size, fill=total_line_color)
        self.canvas.create_line(x1, y_pos, x2, y_pos, fill=total_line_color)
        self.canvas.create_text((x1+x2)/2, y_pos - text_padding, text=f"W = {W:.1f}", font=("Arial", font_size, "bold"), fill=total_line_color, anchor=tk.S)

        # Total Height (H)
        y1, y2 = oy, oy + H * scale
        x_pos = ox - total_dim_offset
        self.canvas.create_line(x_pos - tick_size, y1, x_pos + tick_size, y1, fill=total_line_color)
        self.canvas.create_line(x_pos - tick_size, y2, x_pos + tick_size, y2, fill=total_line_color)
        self.canvas.create_line(x_pos, y1, x_pos, y2, fill=total_line_color)
        self.canvas.create_text(x_pos - text_padding, (y1+y2)/2, text=f"H = {H:.1f}", font=("Arial", font_size, "bold"), fill=total_line_color, anchor=tk.E)

    def parse_rooms(self) -> List[RoomSpec]:
        rooms = []
        for line in self.text.get("1.0", tk.END).strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try: 
                    can_rotate = len(parts) > 3 and parts[3] in ['1', 'yes', 'true', 'rotatable']
                    rooms.append(RoomSpec(parts[0], float(parts[1]), float(parts[2]), can_rotate))
                except ValueError: continue
        return rooms

    def prev(self):
        if self.layouts: self.idx = (self.idx - 1 + len(self.layouts)) % len(self.layouts); self.redraw()
    def next(self):
        if self.layouts: self.idx = (self.idx + 1) % len(self.layouts); self.redraw()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
