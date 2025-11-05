#!/usr/bin/env python3
"""
Final, polished version of the layout generator.
This version features partial placements, connectivity checks, and dynamic entrance finding
to create robust and logical floor plans.
"""

import math
import random
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Tuple, Optional

# -------- Tunables --------
MAX_LAYOUTS = 50
RANDOM_PERM_TRIES = 500
SEED = 24680
CONNECTIVITY_GRID_RES = 1.5

# Collision/geometry tolerances
EPS = 1e-7
ROOM_GAP = 0.0

# -------- Data Structures --------
class RoomSpec:
    def __init__(self, name: str, w: float, h: float):
        self.name, self.w, self.h = name, float(w), float(h)

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
def is_accessible(layout: Layout) -> bool:
    if not layout.entrance_pos: return False

    res = CONNECTIVITY_GRID_RES
    cols = max(1, int(math.ceil(layout.plot_w / res)))
    rows = max(1, int(math.ceil(layout.plot_h / res)))
    
    grid = [[True for _ in range(cols)] for _ in range(rows)]
    free_cell_coords = []
    for r in range(rows):
        for c in range(cols):
            cell_x, cell_y = c * res + res/2, r * res + res/2
            for pr in layout.placed:
                if pr.x <= cell_x < pr.x + pr.w and pr.y <= cell_y < pr.y + pr.h:
                    grid[r][c] = False
                    break
            if grid[r][c]:
                free_cell_coords.append((c, r))

    if not free_cell_coords: return True

    start_c, start_r = int(layout.entrance_pos[0]/res), int(layout.entrance_pos[1]/res)
    start_c, start_r = min(cols-1, max(0, start_c)), min(rows-1, max(0, start_r))
    
    q = [(start_c, start_r)]
    visited = set(q)
    
    while q:
        c, r = q.pop(0)
        if not grid[r][c]: continue
        
        for dc, dr in [(0,1), (0,-1), (1,0), (-1,0)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and grid[nr][nc] and (nc, nr) not in visited:
                visited.add((nc, nr))
                q.append((nc, nr))
    
    return all((c,r) in visited for c, r in free_cell_coords)

class LayoutGenerator:
    def __init__(self, plot_w: float, plot_h: float, rooms: List[RoomSpec]):
        self.W, self.H = plot_w, plot_h
        self.rooms = rooms
        self.rng = random.Random(SEED)

    def generate(self, count: int) -> List[Layout]:
        layouts = []
        seen_sigs = set()
        base_rooms = list(self.rooms)
        
        for _ in range(RANDOM_PERM_TRIES):
            if len(layouts) >= count: break
            self.rng.shuffle(base_rooms)
            
            placed, unplaced = self._place_rooms_greedy(base_rooms)
            layout = Layout(placed, [r.name for r in unplaced], self.W, self.H)
            self._find_and_set_entrance(layout)

            if is_accessible(layout):
                sig = (tuple(sorted((p.name, round(p.x,1), round(p.y,1)) for p in layout.placed)), tuple(sorted(layout.unplaced_names)))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    layouts.append(layout)
        
        return layouts

    def _rects_overlap(self, a: PlacedRoom, b: PlacedRoom) -> bool:
        ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
        bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
        return ax1 < bx2 - EPS and ax2 > bx1 + EPS and ay1 < by2 - EPS and ay2 > by1 + EPS

    def _place_rooms_greedy(self, order: List[RoomSpec]) -> Tuple[List[PlacedRoom], List[RoomSpec]]:
        placed, unplaced = [], []
        for r_spec in order:
            best_cand = None
            candidates = []
            for rotated in (False, True):
                w, h = (r_spec.w, r_spec.h) if not rotated else (r_spec.h, r_spec.w)
                
                test_xs = {0.0, self.W - w}
                test_ys = {0.0, self.H - h}
                for pr in placed:
                    test_xs.update({pr.x + pr.w, pr.x - w})
                    test_ys.update({pr.y + pr.h, pr.y - h})
                
                for x in test_xs:
                    for y_base in [0.0, self.H - h]: candidates.append(PlacedRoom(r_spec.name, x, y_base, w, h, rotated))
                for y in test_ys:
                    for x_base in [0.0, self.W-w]: candidates.append(PlacedRoom(r_spec.name, x_base, y, w, h, rotated))

            self.rng.shuffle(candidates)
            for cand in candidates:
                if self._is_valid_placement(cand, placed):
                    best_cand = cand
                    break
            
            if best_cand: placed.append(best_cand)
            else: unplaced.append(r_spec)
        return placed, unplaced

    def _is_valid_placement(self, cand: PlacedRoom, placed: List[PlacedRoom]) -> bool:
        if cand.x < -EPS or cand.y < -EPS or cand.x + cand.w > self.W + EPS or cand.y + cand.h > self.H + EPS:
            return False
        if not (abs(cand.y)<=EPS or abs(cand.x)<=EPS or abs(cand.x+cand.w-self.W)<=EPS or abs(cand.y+cand.h-self.H)<=EPS):
            return False
        for p in placed:
            if self._rects_overlap(cand, p): return False
        return True

    def _find_and_set_entrance(self, layout: Layout):
        def get_free_segments(total_len: float, occupied: List[Tuple[float, float]]):
            occupied.sort()
            free = []
            last_end = 0.0
            for start, end in occupied:
                if start > last_end + EPS: free.append((last_end, start))
                last_end = max(last_end, end)
            if last_end < total_len - EPS: free.append((last_end, total_len))
            return free

        top_occ = [(p.x, p.x + p.w) for p in layout.placed if abs(p.y) < EPS]
        bot_occ = [(p.x, p.x + p.w) for p in layout.placed if abs(p.y + p.h - self.H) < EPS]
        left_occ = [(p.y, p.y + p.h) for p in layout.placed if abs(p.x) < EPS]
        right_occ = [(p.y, p.y + p.h) for p in layout.placed if abs(p.x + p.w - self.W) < EPS]

        candidates = []
        candidates.extend([('top', seg) for seg in get_free_segments(self.W, top_occ)])
        candidates.extend([('bottom', seg) for seg in get_free_segments(self.W, bot_occ)])
        candidates.extend([('left', seg) for seg in get_free_segments(self.H, left_occ)])
        candidates.extend([('right', seg) for seg in get_free_segments(self.H, right_occ)])
        
        if not candidates: return
        
        best_cand = max(candidates, key=lambda item: item[1][1] - item[1][0])
        edge, (start, end) = best_cand
        mid = (start + end) / 2
        
        if edge == 'top': layout.entrance_pos = (mid, 0.0)
        elif edge == 'bottom': layout.entrance_pos = (mid, self.H)
        elif edge == 'left': layout.entrance_pos = (0.0, mid)
        elif edge == 'right': layout.entrance_pos = (self.W, mid)

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
        self.n_var = tk.IntVar(value=8)
        ttk.Entry(top, width=5, textvariable=self.n_var).pack(side=tk.LEFT, padx=(0,10))
        self.boundary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Boundary", variable=self.boundary_var, command=self.redraw).pack(side=tk.LEFT)
        self.openspace_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Open Space", variable=self.openspace_var, command=self.redraw).pack(side=tk.LEFT)
        ttk.Button(top, text="Generate", command=self.generate).pack(side=tk.RIGHT, padx=6)
        
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.Y, padx=(6,0), pady=6)
        ttk.Label(left, text="Rooms (name w h):").pack(anchor=tk.W)
        self.text = ScrolledText(left, width=30, height=28); self.text.pack(fill=tk.Y)
        self.text.insert("1.0", "R1 6 5\nR2 7 4\nR3 4 4\nR4 5 5\nR5 3 8\nR6 6 6\nR7 3 9\nR8 4 7\nR9 3 3\nR10 4.5 4")
        self.canvas = tk.Canvas(main, bg="white"); self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Button(bottom, text="<< Prev", command=self.prev).pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom, text="Next >>", command=self.next).pack(side=tk.LEFT)
        self.info = ttk.Label(bottom, text="No layouts"); self.info.pack(side=tk.LEFT, padx=10)
        self.status = ttk.Label(bottom, text="Ready"); self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        
        self.layouts, self.idx, self.total_rooms = [], 0, 0
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.generate()

    def generate(self):
        self.status.config(text="Generating...")
        self.root.update_idletasks()
        
        rooms = self.parse_rooms()
        self.total_rooms = len(rooms)
        gen = LayoutGenerator(self.w_var.get(), self.h_var.get(), rooms)
        self.layouts = gen.generate(self.n_var.get())
        self.idx = 0
        
        self.status.config(text=f"Found {len(self.layouts)} valid layouts.")
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts:
            self.info.config(text="No valid layouts found.")
            self.status.config(text="Try a larger plot or fewer rooms.", foreground="red")
            return

        layout = self.layouts[self.idx]
        pad, W, H = 20, layout.plot_w, layout.plot_h
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        scale = min((cw - 2*pad) / W, (ch - 2*pad) / H) if W>0 and H>0 else 1
        ox, oy = pad, pad

        if self.openspace_var.get():
            self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, fill="#d0d0d0", outline="")
        
        for r in layout.placed:
            x1, y1 = ox+r.x*scale, oy+r.y*scale
            x2, y2 = x1+r.w*scale, y1+r.h*scale
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4ea3ff", outline="#003366", width=1.5)
            self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f"{r.name}\n{r.w:.1f}x{r.h:.1f}", fill="white", font=("Arial",9))
        
        if self.boundary_var.get():
            self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, width=2, outline="#333")
            
        if layout.entrance_pos:
            ex, ey = ox + layout.entrance_pos[0]*scale, oy + layout.entrance_pos[1]*scale
            is_vertical = abs(layout.entrance_pos[0] - 0) < EPS or abs(layout.entrance_pos[0] - W) < EPS
            if is_vertical: self.canvas.create_line(ex, ey-10, ex, ey+10, fill="red", width=5)
            else: self.canvas.create_line(ex-10, ey, ex+10, ey, fill="red", width=5)

        self.info.config(text=f"Layout {self.idx+1}/{len(self.layouts)} — Placed {layout.placed_count}/{self.total_rooms}")
        status_text = "All rooms placed." if not layout.unplaced_names else f"Unplaced: {', '.join(layout.unplaced_names)}"
        self.status.config(text=status_text, foreground="green" if not layout.unplaced_names else "red")
        
    def parse_rooms(self) -> List[RoomSpec]:
        rooms = []
        for line in self.text.get("1.0", tk.END).strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try: rooms.append(RoomSpec(parts[0], float(parts[1]), float(parts[2])))
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
