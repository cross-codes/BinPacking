#!/usr/bin/env python3
"""
Boundary-hugging backtracking layouts (NO OVERLAP) + optional routed corridors.
This version supports partial layouts if not all rooms can be placed.
Tkinter-only, single file. Run:

    python3 single_layout_app_boundary_bt.py
"""

import math
import heapq
import random
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Tuple, Set

# -------- Tunables --------
MAX_LAYOUTS = 50
BT_NODE_LIMIT = 100_000 # Reduced to prevent long hangs on impossible layouts
RANDOM_PERM_TRIES = 500
SEED = 24680

# Collision/geometry tolerances
EPS = 1e-7
EDGE_GAP = 0.0
ROOM_GAP = 0.0

# -------- Data --------
class RoomSpec:
    def __init__(self, name: str, w: float, h: float):
        self.name = name
        self.w = float(w)
        self.h = float(h)

class PlacedRoom:
    def __init__(self, name: str, x: float, y: float, w: float, h: float, edge: str, rotated: bool):
        self.name = name
        self.x = float(x); self.y = float(y)
        self.w = float(w); self.h = float(h)
        self.edge = edge
        self.rotated = bool(rotated)

class Layout:
    def __init__(self, placed: List[PlacedRoom], unplaced_names: List[str], plot_w: float, plot_h: float):
        self.placed = placed
        self.unplaced_names = unplaced_names
        self.plot_w = plot_w; self.plot_h = plot_h
        self.placed_count = len(placed)
        self.rooms_area = sum(p.w * p.h for p in placed)

# -------- Geometry helpers --------
def rects_overlap(a: PlacedRoom, b: PlacedRoom) -> bool:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ax1 += ROOM_GAP*0.5; ay1 += ROOM_GAP*0.5
    ax2 -= ROOM_GAP*0.5; ay2 -= ROOM_GAP*0.5
    bx1 += ROOM_GAP*0.5; by1 += ROOM_GAP*0.5
    bx2 -= ROOM_GAP*0.5; by2 -= ROOM_GAP*0.5
    return (ax1 < bx2 - EPS and ax2 > bx1 + EPS and ay1 < by2 - EPS and ay2 > by1 + EPS)

def collides_any(candidate: PlacedRoom, placed: List[PlacedRoom]) -> bool:
    for pr in placed:
        if rects_overlap(candidate, pr):
            return True
    return False

# -------- Boundary-hugging backtracker --------
class BoundaryBacktracker:
    def __init__(self, plot_w: float, plot_h: float, rooms: List[RoomSpec]):
        self.W = plot_w
        self.H = plot_h
        self.rooms = rooms
        self.rng = random.Random(SEED)

    def try_place_all(self, order: List[RoomSpec]) -> Layout:
        placed: List[PlacedRoom] = []
        unplaced_names: List[str] = []
        
        def fits(candidate: PlacedRoom, current_placed: List[PlacedRoom]) -> bool:
            if candidate.x < -EPS or candidate.y < -EPS: return False
            if candidate.x + candidate.w > self.W + EPS: return False
            if candidate.y + candidate.h > self.H + EPS: return False
            
            touches = (
                abs(candidate.y) <= max(EPS, EDGE_GAP) or
                abs(candidate.x) <= max(EPS, EDGE_GAP) or
                abs(candidate.x + candidate.w - self.W) <= max(EPS, EDGE_GAP) or
                abs(candidate.y + candidate.h - self.H) <= max(EPS, EDGE_GAP)
            )
            if not touches: return False
            
            return not collides_any(candidate, current_placed)

        for r in order:
            found_spot = False
            node_count = 0
            
            all_candidates = []
            for rotated in (False, True):
                w, h = (r.w, r.h) if not rotated else (r.h, r.w)
                test_xs = {0.0, self.W - w}
                test_ys = {0.0, self.H - h}
                for pr in placed:
                    test_xs.add(pr.x + pr.w + ROOM_GAP)
                    test_xs.add(pr.x - w - ROOM_GAP)
                    test_ys.add(pr.y + pr.h + ROOM_GAP)
                    test_ys.add(pr.y - h - ROOM_GAP)

                for x in test_xs:
                    all_candidates.append(PlacedRoom(r.name, x, 0.0, w, h, 'top', rotated))
                    all_candidates.append(PlacedRoom(r.name, x, self.H - h, w, h, 'bottom', rotated))
                for y in test_ys:
                    all_candidates.append(PlacedRoom(r.name, 0.0, y, w, h, 'left', rotated))
                    all_candidates.append(PlacedRoom(r.name, self.W - w, y, w, h, 'right', rotated))
            
            self.rng.shuffle(all_candidates)

            for cand in all_candidates:
                if node_count > BT_NODE_LIMIT / len(order): break
                node_count += 1
                if fits(cand, placed):
                    placed.append(cand)
                    found_spot = True
                    break
            
            if not found_spot:
                unplaced_names.append(r.name)

        return Layout(placed, unplaced_names, self.W, self.H)

    def generate(self, count: int) -> List[Layout]:
        layouts: List[Layout] = []
        seen_sigs = set()
        
        master_rng = random.Random(SEED)
        base_rooms = list(self.rooms)
        tries = 0
        while len(layouts) < count and tries < RANDOM_PERM_TRIES:
            tries += 1
            master_rng.shuffle(base_rooms)
            
            layout = self.try_place_all(base_rooms)
            
            # Signature now includes unplaced rooms to ensure variety
            sig = (
                tuple(sorted((p.name, round(p.x,2), round(p.y,2)) for p in layout.placed)),
                tuple(sorted(layout.unplaced_names))
            )

            if sig not in seen_sigs:
                seen_sigs.add(sig)
                layouts.append(layout)
        
        # If no layouts were found at all, return the best effort from one try
        if not layouts and self.rooms:
            layouts.append(self.try_place_all(self.rooms))

        return layouts[:count]

# -------- UI --------
SAMPLE = """R1 6 5
R2 7 4
R3 4 4
R4 5 5
R5 3 8
R6 6 6
R7 3 9
R8 4 7
R9 3 3
R10 4.5 4
"""

CANVAS_W, CANVAS_H = 1000, 720

class App:
    def __init__(self, root):
        self.root = root
        root.title("Boundary Layouts with Partial Placement")

        top = ttk.Frame(root); top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Label(top, text="Plot width:").pack(side=tk.LEFT)
        self.w_var = tk.DoubleVar(value=20.0)
        ttk.Entry(top, width=7, textvariable=self.w_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Plot height:").pack(side=tk.LEFT)
        self.h_var = tk.DoubleVar(value=25.0)
        ttk.Entry(top, width=7, textvariable=self.h_var).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Layouts:").pack(side=tk.LEFT, padx=(12,0))
        self.n_var = tk.IntVar(value=8)
        ttk.Entry(top, width=5, textvariable=self.n_var).pack(side=tk.LEFT, padx=4)

        self.boundary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Show boundary", variable=self.boundary_var, command=lambda: self.redraw()).pack(side=tk.LEFT, padx=8)

        self.corridor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Show open space", variable=self.corridor_var, command=lambda: self.redraw()).pack(side=tk.LEFT, padx=8)

        self.entrance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Mark entrance", variable=self.entrance_var, command=lambda: self.redraw()).pack(side=tk.LEFT, padx=8)

        ttk.Button(top, text="Generate", command=self.generate).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load sample", command=self.load_sample).pack(side=tk.LEFT)

        main = ttk.Frame(root); main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        ttk.Label(left, text="Rooms (name w h  OR  w h):").pack(anchor=tk.W)
        self.text = ScrolledText(left, width=30, height=28); self.text.pack(fill=tk.Y)
        self.text.insert("1.0", SAMPLE)

        right = ttk.Frame(main); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root); controls.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        ttk.Button(controls, text="<< Prev", command=self.prev_layout).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Next >>", command=self.next_layout).pack(side=tk.LEFT)
        self.info = ttk.Label(controls, text="No layouts yet"); self.info.pack(side=tk.LEFT, padx=10)
        self.status = ttk.Label(root, text="Ready", anchor=tk.W); self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        
        self.total_rooms = 0
        self.layouts: List[Layout] = []
        self.idx = 0

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.generate()

    def load_sample(self):
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", SAMPLE)

    def parse_rooms(self, s: str) -> List[RoomSpec]:
        out=[]; cnt=1
        for line in s.strip().splitlines():
            parts=[p for p in line.replace(',', ' ').split() if p]
            if not parts: continue
            if len(parts)==2:
                w=float(parts[0]); h=float(parts[1])
                out.append(RoomSpec(f"R{cnt}", w, h)); cnt+=1
            elif len(parts)>=3:
                out.append(RoomSpec(parts[0], float(parts[1]), float(parts[2])))
        self.total_rooms = len(out)
        return out

    def generate(self):
        self.status.config(text="Generating...")
        self.root.update_idletasks()
        try:
            W = float(self.w_var.get()); H = float(self.h_var.get())
        except Exception as e:
            self.status.config(text=f"Bad plot size: {e}"); return
        rooms = self.parse_rooms(self.text.get("1.0", tk.END))
        if not rooms:
            self.status.config(text="No rooms"); return
        n = max(1, min(int(self.n_var.get()), MAX_LAYOUTS))

        bt = BoundaryBacktracker(W, H, rooms)
        self.layouts = bt.generate(n)
        self.idx = 0
        
        self.status.config(text=f"Generated {len(self.layouts)} layouts.")
        self.redraw()

    def prev_layout(self):
        if not self.layouts: return
        self.idx = (self.idx - 1) % len(self.layouts); self.redraw()

    def next_layout(self):
        if not self.layouts: return
        self.idx = (self.idx + 1) % len(self.layouts); self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts:
            self.info.config(text="No layouts yet")
            self.status.config(text="No layouts found. Try a larger plot.")
            return

        L = self.layouts[self.idx]
        W,H = L.plot_w, L.plot_h

        pad=20
        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        scale = min((cw-2*pad)/W, (ch-2*pad)/H) if W>0 and H>0 else 1.0
        ox, oy = pad, pad
        
        if self.corridor_var.get():
            self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, fill="#d0d0d0", outline="")
        
        for pr in L.placed:
            x1=ox+pr.x*scale; y1=oy+pr.y*scale
            x2=x1+pr.w*scale; y2=y1+pr.h*scale
            self.canvas.create_rectangle(x1,y1,x2,y2, fill="#4ea3ff", outline="#003366", width=1)
            cx=(x1+x2)/2; cy=(y1+y2)/2
            self.canvas.create_text(cx, cy, text=f"{pr.name}\n{pr.w:.1f}x{pr.h:.1f}", font=("Arial", 9), fill="white")
        
        if self.boundary_var.get():
            self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, width=2, outline="#222222")

        if self.entrance_var.get():
            self.mark_entrance(L, ox, oy, scale)

        info_text = f"Layout {self.idx+1}/{len(self.layouts)} — Placed {L.placed_count}/{self.total_rooms}, Area {L.rooms_area:.1f}"
        self.info.config(text=info_text)

        if L.unplaced_names:
            unplaced_str = ", ".join(L.unplaced_names)
            self.status.config(text=f"Unplaced rooms: {unplaced_str}", foreground="red")
        else:
            self.status.config(text=f"All {self.total_rooms} rooms placed successfully.", foreground="green")

    def mark_entrance(self, L: Layout, ox: float, oy: float, scale: float):
        W, H = L.plot_w, L.plot_h
        
        def update_free_intervals(free_intervals, occupied_interval):
            next_free = []
            occ_s, occ_e = occupied_interval
            for free_s, free_e in free_intervals:
                if free_e < occ_s + EPS or free_s > occ_e - EPS:
                    next_free.append((free_s, free_e)); continue
                if free_s < occ_s - EPS: next_free.append((free_s, occ_s))
                if free_e > occ_e + EPS: next_free.append((occ_e, free_e))
            return next_free

        top_free, bottom_free, left_free, right_free = [(0, W)], [(0, W)], [(0, H)], [(0, H)]

        for pr in L.placed:
            if abs(pr.y) < EPS: top_free = update_free_intervals(top_free, (pr.x, pr.x + pr.w))
            if abs(pr.y + pr.h - H) < EPS: bottom_free = update_free_intervals(bottom_free, (pr.x, pr.x + pr.w))
            if abs(pr.x) < EPS: left_free = update_free_intervals(left_free, (pr.y, pr.y + pr.h))
            if abs(pr.x + pr.w - W) < EPS: right_free = update_free_intervals(right_free, (pr.y, pr.y + pr.h))

        possible_entrances = []
        min_opening = 1.0
        for s, e in top_free:
            if e - s > min_opening: possible_entrances.append(('top', s, e))
        for s, e in bottom_free:
            if e - s > min_opening: possible_entrances.append(('bottom', s, e))
        for s, e in left_free:
            if e - s > min_opening: possible_entrances.append(('left', s, e))
        for s, e in right_free:
            if e - s > min_opening: possible_entrances.append(('right', s, e))
        
        if not possible_entrances: return
        
        rng = random.Random(self.idx); edge, start, end = rng.choice(possible_entrances)
        mid = (start + end) / 2
        marker_len = min(end - start, 2.0) * 0.8
        
        if edge == 'top':
            x1, y1, x2, y2 = ox+(mid-marker_len/2)*scale, oy, ox+(mid+marker_len/2)*scale, oy
        elif edge == 'bottom':
            x1, y1, x2, y2 = ox+(mid-marker_len/2)*scale, oy+H*scale, ox+(mid+marker_len/2)*scale, oy+H*scale
        elif edge == 'left':
            x1, y1, x2, y2 = ox, oy+(mid-marker_len/2)*scale, ox, oy+(mid+marker_len/2)*scale
        else: # right
            x1, y1, x2, y2 = ox+W*scale, oy+(mid-marker_len/2)*scale, ox+W*scale, oy+(mid+marker_len/2)*scale
        self.canvas.create_line(x1, y1, x2, y2, fill="red", width=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
