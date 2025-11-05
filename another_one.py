#!/usr/bin/env python3
"""
Boundary-hugging backtracking layouts (NO OVERLAP) + optional routed corridors.
This version uses a fully randomized room order to generate more varied layouts.
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
GRID_RESOLUTION = 1.0        # grid for routing
MAX_LAYOUTS = 50
BT_NODE_LIMIT = 250_000
RANDOM_PERM_TRIES = 500 # Increased tries for better variety
SEED = 24680

# Collision/geometry tolerances
EPS = 1e-7                   # numerical tolerance
EDGE_GAP = 0.0               # keep 0 to ensure rooms still "touch" boundary
ROOM_GAP = 0.0               # interior gap between rooms (set >0 to enforce tiny separation)

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

class CorridorRect:
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = float(x); self.y = float(y)
        self.w = float(w); self.h = float(h)

class Layout:
    def __init__(self, placed: List[PlacedRoom], plot_w: float, plot_h: float, corridors: List[CorridorRect]):
        self.placed = placed
        self.plot_w = plot_w; self.plot_h = plot_h
        self.corridors = corridors
        self.placed_count = len(placed)
        self.rooms_area = sum(p.w * p.h for p in placed)

# -------- Geometry helpers --------
def rects_overlap(a: PlacedRoom, b: PlacedRoom) -> bool:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    # Apply small gap if desired
    ax1 += ROOM_GAP*0.5; ay1 += ROOM_GAP*0.5
    ax2 -= ROOM_GAP*0.5; ay2 -= ROOM_GAP*0.5
    bx1 += ROOM_GAP*0.5; by1 += ROOM_GAP*0.5
    bx2 -= ROOM_GAP*0.5; by2 -= ROOM_GAP*0.5
    # Proper overlap (not just touching edge)
    return (ax1 < bx2 - EPS and ax2 > bx1 + EPS and ay1 < by2 - EPS and ay2 > by1 + EPS)

def collides_any(candidate: PlacedRoom, placed: List[PlacedRoom]) -> bool:
    for pr in placed:
        if rects_overlap(candidate, pr):
            return True
    return False

# -------- Routing helpers --------
def build_grid(plot_w: float, plot_h: float, res: float):
    cols = max(1, int(math.ceil(plot_w / res)))
    rows = max(1, int(math.ceil(plot_h / res)))
    return cols, rows

def occupancy_from_rooms(placed: List[PlacedRoom], cols: int, rows: int, res: float):
    grid = [[False for _ in range(cols)] for _ in range(rows)]
    for pr in placed:
        x0 = int(math.floor(pr.x / res))
        y0 = int(math.floor(pr.y / res))
        x1 = int(math.ceil((pr.x + pr.w) / res))
        y1 = int(math.ceil((pr.y + pr.h) / res))
        for r in range(max(0,y0), min(rows, y1)):
            for c in range(max(0,x0), min(cols, x1)):
                grid[r][c] = True
    return grid

def point_to_cell(px: float, py: float, res: float):
    return int(px // res), int(py // res)

def access_point_for_room_boundary(pr: PlacedRoom) -> Tuple[float,float]:
    # Corridor access on the side facing into the interior
    if pr.edge == 'top':    return pr.x + pr.w/2.0, pr.y + pr.h
    if pr.edge == 'bottom': return pr.x + pr.w/2.0, pr.y
    if pr.edge == 'left':   return pr.x + pr.w,     pr.y + pr.h/2.0
    return pr.x, pr.y + pr.h/2.0  # right

def astar(start, goal, grid):
    rows = len(grid); cols = len(grid[0]) if rows>0 else 0
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    def neighbors(c):
        x,y = c
        for dx,dy in ((1,0),(-1,0),(0,1),(-1,0)):
            nx,ny = x+dx,y+dy
            if 0<=nx<cols and 0<=ny<rows and not grid[ny][nx]:
                yield (nx,ny)
    openh=[]; heapq.heappush(openh,(h(start,goal),0,start))
    came={start:None}; g={start:0}; closed=set()
    while openh:
        _,cost,cur = heapq.heappop(openh)
        if cur == goal:
            path=[]; c=cur
            while c is not None:
                path.append(c); c=came[c]
            path.reverse(); return path
        if cur in closed: continue
        closed.add(cur)
        for nb in neighbors(cur):
            tg = g[cur] + 1
            if nb in g and tg >= g[nb]: continue
            came[nb] = cur; g[nb] = tg
            heapq.heappush(openh, (tg + h(nb,goal), tg, nb))
    return []

def cells_to_rects(cells: Set[Tuple[int,int]], res: float) -> List[CorridorRect]:
    if not cells: return []
    todo = set(cells); rects=[]
    while todo:
        s = next(iter(todo))
        stack=[s]; comp=set()
        while stack:
            c = stack.pop()
            if c not in todo: continue
            todo.remove(c); comp.add(c)
            x,y = c
            for dx,dy in ((1,0),(-1,0),(0,1),(-1,0)):
                nb=(x+dx,y+dy)
                if nb in todo: stack.append(nb)
        xs=[c[0] for c in comp]; ys=[c[1] for c in comp]
        minx=min(xs); maxx=max(xs); miny=min(ys); maxy=max(ys)
        rects.append(CorridorRect(minx*res, miny*res, (maxx-minx+1)*res, (maxy-miny+1)*res))
    return rects

# -------- UPDATED Boundary-hugging backtracker --------
class BoundaryBacktracker:
    """
    Places rooms by trying all valid boundary-touching positions for each room.
    This version uses a fully randomized room order to generate more varied layouts.
    """
    def __init__(self, plot_w: float, plot_h: float, rooms: List[RoomSpec]):
        self.W = plot_w
        self.H = plot_h
        self.rooms = rooms
        self.rng = random.Random(SEED)

    def try_place_all(self, order: List[RoomSpec], want: int) -> List[Layout]:
        layouts: List[Layout] = []
        seen = set()
        node_count = 0

        def signature(placed: List[PlacedRoom]):
            return tuple(sorted((p.name, round(p.x, 2), round(p.y, 2), round(p.w, 2), round(p.h, 2)) for p in placed))

        def fits(candidate: PlacedRoom, placed: List[PlacedRoom]) -> bool:
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
            
            return not collides_any(candidate, placed)

        def backtrack(i: int, placed: List[PlacedRoom]):
            nonlocal node_count
            if len(layouts) >= want: return
            if node_count > BT_NODE_LIMIT: return
            node_count += 1

            if i == len(order):
                final_placed = []
                for p in placed:
                    edge = 'unknown'
                    if abs(p.y) < EPS: edge = 'top'
                    elif abs(p.y + p.h - self.H) < EPS: edge = 'bottom'
                    elif abs(p.x) < EPS: edge = 'left'
                    elif abs(p.x + p.w - self.W) < EPS: edge = 'right'
                    final_placed.append(PlacedRoom(p.name, p.x, p.y, p.w, p.h, edge, p.rotated))
                
                sig = signature(final_placed)
                if sig in seen: return
                seen.add(sig)
                
                layouts.append(Layout(list(final_placed), self.W, self.H, [])) # Corridors can be slow, compute later
                return

            r = order[i]
            
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
            
            tested_positions = set()
            for cand in all_candidates:
                pos_sig = (round(cand.x, 2), round(cand.y, 2), round(cand.w, 2), round(cand.h, 2))
                if pos_sig in tested_positions: continue
                tested_positions.add(pos_sig)

                if fits(cand, placed):
                    placed.append(cand)
                    backtrack(i + 1, placed)
                    placed.pop()
                    if len(layouts) >= want: return

        backtrack(0, [])
        return layouts

    def generate(self, count: int) -> List[Layout]:
        layouts: List[Layout] = []
        seen_sigs = set()
        
        def add_unique(ls: List[Layout]):
            nonlocal layouts, seen_sigs
            initial_count = len(layouts)
            for L in ls:
                sig = tuple(sorted((p.name, round(p.x,2), round(p.y,2), round(p.w,2), round(p.h,2)) for p in L.placed))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    layouts.append(L)
                    if len(layouts) >= count:
                        return True
            return len(layouts) > initial_count
            
        # --- KEY CHANGE: Use fully randomized ordering ---
        master_rng = random.Random(SEED)
        base_rooms = list(self.rooms)
        tries = 0
        while len(layouts) < count and tries < RANDOM_PERM_TRIES:
            tries += 1
            master_rng.shuffle(base_rooms)
            order = list(base_rooms)

            # Try to find one new layout with this order
            batch = self.try_place_all(order, 1)
            add_unique(batch)

        # Post-process to add corridors if enabled
        for l in layouts:
            l.corridors = self.route_corridors(l.placed)

        return layouts[:count]

    # ----- Corridors (optional) -----
    def route_corridors(self, placed: List[PlacedRoom]) -> List[CorridorRect]:
        if not placed: return []
        res = GRID_RESOLUTION
        cols, rows = build_grid(self.W, self.H, res)
        occ = occupancy_from_rooms(placed, cols, rows, res)

        access = []
        for pr in placed:
            ax, ay = access_point_for_room_boundary(pr)
            start = point_to_cell(ax, ay, res)
            q=[start]; seen=set(q); found=None
            while q and found is None:
                c = q.pop(0); x,y = c
                if 0<=x<cols and 0<=y<rows and not occ[y][x]:
                    found=c; break
                for dx,dy in ((1,0),(-1,0),(0,1),(-1,0)):
                    nb=(x+dx,y+dy)
                    if nb not in seen and 0<=nb[0]<cols and 0<=nb[1]<rows:
                        seen.add(nb); q.append(nb)
            if found is None:
                found=(min(max(0,start[0]),cols-1), min(max(0,start[1]),rows-1))
            access.append(found)

        def prim(points):
            n=len(points)
            if n<=1: return []
            in_m=[False]*n; dist=[float('inf')]*n; par=[-1]*n; dist[0]=0.0
            edges=[]
            for _ in range(n):
                v=-1; best=float('inf')
                for i in range(n):
                    if not in_m[i] and dist[i]<best:
                        best=dist[i]; v=i
                if v==-1: break
                in_m[v]=True
                if par[v]!=-1: edges.append((v,par[v]))
                for u in range(n):
                    if in_m[u]: continue
                    d = abs(points[v][0]-points[u][0]) + abs(points[v][1]-points[u][1])
                    if d<dist[u]: dist[u]=d; par[u]=v
            return edges

        edges = prim(access)
        paths=[]
        for i,j in edges:
            s=access[i]; g=access[j]
            p = astar(s,g,occ)
            if p:
                paths.append(p)
            else:
                inter1=(s[0], g[1]); inter2=(g[0], s[1])
                p1=astar(s, inter1, occ); p2=astar(inter1, g, occ)
                if p1 and p2: paths.append(p1+p2[1:]); continue
                p1=astar(s, inter2, occ); p2=astar(inter2, g, occ)
                if p1 and p2: paths.append(p1+p2[1:])
        cells=set()
        for p in paths:
            for c in p: cells.add(c)
        return cells_to_rects(cells, res)


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
        root.title("Boundary Backtracking Layouts (no-overlap)")

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
        ttk.Checkbutton(top, text="Show corridors", variable=self.corridor_var, command=lambda: self.redraw()).pack(side=tk.LEFT, padx=8)

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

        controls = ttk.Frame(root); controls.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(controls, text="<< Prev", command=self.prev_layout).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Next >>", command=self.next_layout).pack(side=tk.LEFT)
        self.info = ttk.Label(controls, text="No layouts yet"); self.info.pack(side=tk.LEFT, padx=10)
        self.status = ttk.Label(root, text="Ready"); self.status.pack(side=tk.BOTTOM, fill=tk.X)

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
        if not self.layouts:
            self.status.config(text="No layouts found (try larger plot or fewer/lower rooms)")
            self.canvas.delete("all")
            self.info.config(text="No layouts yet")
            return
        self.status.config(text=f"Generated {len(self.layouts)} layouts")
        self.redraw()

    def prev_layout(self):
        if not self.layouts: return
        self.idx = (self.idx - 1) % len(self.layouts); self.redraw()

    def next_layout(self):
        if not self.layouts: return
        self.idx = (self.idx + 1) % len(self.layouts); self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts: return
        L = self.layouts[self.idx]
        W,H = L.plot_w, L.plot_h

        pad=20
        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        scale = min((cw-2*pad)/W, (ch-2*pad)/H) if W>0 and H>0 else 1.0
        ox, oy = pad, pad
        
        if self.corridor_var.get():
            self.canvas.create_rectangle(ox, oy, ox+W*scale, oy+H*scale, fill="#d0d0d0", outline="")
            for crect in L.corridors:
                x1=ox+crect.x*scale; y1=oy+crect.y*scale
                x2=x1+crect.w*scale; y2=y1+crect.h*scale
                self.canvas.create_rectangle(x1,y1,x2,y2, fill="gray", outline="")


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

        self.info.config(text=f"Layout {self.idx+1}/{len(self.layouts)} — rooms {L.placed_count}, area {L.rooms_area:.1f}")

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
