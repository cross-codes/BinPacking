#!/usr/bin/env python3
"""
Final, stable version of the layout generator.
This version uses a robust greedy placement algorithm with integrated connectivity checks
to ensure all generated layouts are valid and accessible.
"""

import math
import random
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Tuple, Optional, Set

# -------- Tunables --------
MAX_LAYOUTS = 50
RANDOM_PERM_TRIES = 800 # Increased tries for the robust checker
SEED = 24680
CONNECTIVITY_GRID_RES = 1.5
EPS = 1e-7

# -------- Data Structures --------
class RoomSpec:
    def __init__(self, name: str, w: float, h: float, can_rotate: bool):
        self.name, self.w, self.h, self.can_rotate = name, float(w), float(h), can_rotate

class PlacedRoom:
    def __init__(self, name: str, x: float, y: float, w: float, h: float, rotated: bool):
        self.name, self.x, self.y, self.w, self.h, self.rotated = name, float(x), float(y), float(w), float(h), bool(rotated)

class Layout:
    def __init__(self, placed: List[PlacedRoom], unplaced: List[str], plot_w: float, plot_h: float):
        self.placed, self.unplaced_names = placed, unplaced
        self.plot_w, self.plot_h = plot_w, plot_h
        self.placed_count = len(placed)
        self.entrance_pos: Optional[Tuple[float, float]] = None

# -------- Core Logic --------
class LayoutGenerator:
    def __init__(self, plot_w: float, plot_h: float, rooms: List[RoomSpec]):
        self.W, self.H, self.rooms, self.rng = plot_w, plot_h, rooms, random.Random(SEED)

    def generate(self, count: int) -> List[Layout]:
        layouts, seen_sigs = [], set()
        base_rooms = list(self.rooms)
        
        for _ in range(RANDOM_PERM_TRIES):
            if len(layouts) >= count: break
            self.rng.shuffle(base_rooms)
            
            placed, unplaced = self._place_rooms_greedy(base_rooms)
            if not placed: continue

            layout = Layout(placed, [r.name for r in unplaced], self.W, self.H)
            self._find_and_set_entrance(layout)
            
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
                    best_cand = cand_room
                    break
            
            if best_cand: placed.append(best_cand)
            else: unplaced.append(r_spec)
        return placed, unplaced

    def _get_placement_candidates(self, w: float, h: float, placed: List[PlacedRoom]) -> List[Tuple[float, float]]:
        # Generates candidate (x, y) positions for a new room.
        test_xs, test_ys = {0.0, self.W - w}, {0.0, self.H - h}
        for pr in placed:
            test_xs.update({pr.x + pr.w, pr.x - w})
            test_ys.update({pr.y + pr.h, pr.y - h})
        
        candidates = set()
        for x in test_xs: candidates.update([(x, 0.0), (x, self.H - h)])
        for y in test_ys: candidates.update([(0.0, y), (self.W - w, y)])
        return list(candidates)

    def _is_valid_move(self, cand: PlacedRoom, placed: List[PlacedRoom]) -> bool:
        # 1. Basic boundary and overlap checks
        if not (cand.x >= -EPS and cand.y >= -EPS and cand.x + cand.w <= self.W + EPS and cand.y + cand.h <= self.H + EPS): return False
        if not (abs(cand.y)<=EPS or abs(cand.x)<=EPS or abs(cand.x+cand.w-self.W)<=EPS or abs(cand.y+cand.h-self.H)<=EPS): return False
        if any(self._rects_overlap(cand, p) for p in placed): return False

        # 2. Full connectivity check
        temp_placed = placed + [cand]
        potential_entrances = self._get_potential_entrances(temp_placed)
        if not potential_entrances: return False

        return self._is_accessible_from_any_entrance(temp_placed, potential_entrances)

    def _is_accessible_from_any_entrance(self, placed: List[PlacedRoom], entrances: List[Tuple[float,float]]) -> bool:
        res = CONNECTIVITY_GRID_RES
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

        start_node = None
        for ex, ey in entrances:
            c, r = min(cols-1, max(0, int(ex/res))), min(rows-1, max(0, int(ey/res)))
            if grid[r][c]:
                start_node = (c, r); break
        if start_node is None: return False

        q, visited, count = [start_node], {start_node}, 0
        while q:
            c, r = q.pop(0)
            count += 1
            for dc, dr in [(0,1), (0,-1), (1,0), (-1,0)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < cols and 0 <= nr < rows and grid[nr][nc] and (nc, nr) not in visited:
                    visited.add((nc, nr)); q.append((nc, nr))
        
        return count == total_free_cells

    def _get_potential_entrances(self, placed: List[PlacedRoom]) -> List[Tuple[float,float]]:
        entrances = []
        bounds = self._get_free_boundary_segments(placed)
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
        return {
            'top': get_segs(self.W, [(p.x, p.x + p.w) for p in placed if abs(p.y) < EPS]),
            'bottom': get_segs(self.W, [(p.x, p.x + p.w) for p in placed if abs(p.y + p.h - self.H) < EPS]),
            'left': get_segs(self.H, [(p.y, p.y + p.h) for p in placed if abs(p.x) < EPS]),
            'right': get_segs(self.H, [(p.y, p.y + p.h) for p in placed if abs(p.x + p.w - self.W) < EPS]),
        }

    def _find_and_set_entrance(self, layout: Layout):
        bounds = self._get_free_boundary_segments(layout.placed)
        candidates = [(edge, seg) for edge, segs in bounds.items() for seg in segs if seg[1] - seg[0] > 1.0]
        if not candidates: return
        edge, (start, end) = max(candidates, key=lambda item: item[1][1] - item[1][0])
        mid = (start + end) / 2
        pos_map = {'top': (mid, 0.0), 'bottom': (mid, self.H), 'left': (0.0, mid), 'right': (self.W, mid)}
        layout.entrance_pos = pos_map.get(edge)

    def _rects_overlap(self, a: PlacedRoom, b: PlacedRoom) -> bool:
        return a.x < b.x + b.w - EPS and a.x + a.w > b.x + EPS and \
               a.y < b.y + b.h - EPS and a.y + a.h > b.y + EPS

# -------- UI --------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Intelligent Layout Generator")
        
        top = ttk.Frame(root); top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        main = ttk.Frame(root); main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        bottom = ttk.Frame(root); bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        
        ttk.Label(top, text="Plot W:").pack(side=tk.LEFT)
        self.w_var = tk.DoubleVar(value=20.0)
        ttk.Entry(top, width=7, textvariable=self.w_var).pack(side=tk.LEFT, padx=(0,4))
        ttk.Label(top, text="H:").pack(side=tk.LEFT)
        self.h_var = tk.DoubleVar(value=25.0)
        ttk.Entry(top, width=7, textvariable=self.h_var).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(top, text="Layouts:").pack(side=tk.LEFT)
        self.n_var = tk.IntVar(value=10)
        ttk.Entry(top, width=5, textvariable=self.n_var).pack(side=tk.LEFT, padx=(0,10))
        self.boundary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Boundary", variable=self.boundary_var, command=self.redraw).pack(side=tk.LEFT)
        self.openspace_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Open Space", variable=self.openspace_var, command=self.redraw).pack(side=tk.LEFT)
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
        self.status.config(text="Generating..."); self.root.update_idletasks()
        rooms = self.parse_rooms(); self.total_rooms = len(rooms)
        if not rooms: self.status.config(text="No rooms specified."); return
        gen = LayoutGenerator(self.w_var.get(), self.h_var.get(), rooms)
        self.layouts = gen.generate(self.n_var.get()); self.idx = 0
        self.status.config(text=f"Found {len(self.layouts)} valid layouts."); self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts:
            self.info.config(text="No valid layouts."); self.status.config(text="Try larger plot or fewer rooms.", foreground="red"); return

        layout = self.layouts[self.idx]
        pad, W, H = 20, layout.plot_w, layout.plot_h
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        scale = min((cw - 2*pad) / W, (ch - 2*pad) / H) if W*H>0 else 1
        ox, oy = pad, pad

        if self.openspace_var.get(): self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, fill="#d0d0d0", outline="")
        
        for r in layout.placed:
            x1, y1, x2, y2 = ox+r.x*scale, oy+r.y*scale, ox+(r.x+r.w)*scale, oy+(r.y+r.h)*scale
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4ea3ff", outline="#003366", width=1.5)
            name_tag = f"{r.name}*" if r.rotated else r.name
            self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f"{name_tag}\n{r.w:.1f}x{r.h:.1f}", fill="white", font=("Arial",9))
        
        if self.boundary_var.get(): self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, width=2, outline="#333")
            
        if layout.entrance_pos:
            ex, ey = ox + layout.entrance_pos[0]*scale, oy + layout.entrance_pos[1]*scale
            is_vertical = abs(layout.entrance_pos[0]) < EPS or abs(layout.entrance_pos[0] - W) < EPS
            dx, dy = (0, 10) if is_vertical else (10, 0)
            self.canvas.create_line(ex-dx, ey-dy, ex+dx, ey+dy, fill="red", width=5)

        self.info.config(text=f"Layout {self.idx+1}/{len(self.layouts)} — Placed {layout.placed_count}/{self.total_rooms}")
        status_text = "All rooms placed." if not layout.unplaced_names else f"Unplaced: {', '.join(layout.unplaced_names)}"
        self.status.config(text=status_text, foreground="green" if not layout.unplaced_names else "red")
        
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
