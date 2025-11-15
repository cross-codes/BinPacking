import tkinter as tk
from tkinter import ttk, messagebox
import threading, random, math, copy
from collections import deque


def clamp(v, a, b):
    return max(a, min(b, v))


class Room:
    def __init__(self, w, h, idx, name=None, fixed=False):
        self.w0 = int(w)
        self.h0 = int(h)
        self.idx = idx
        self.name = name if name else f"R{idx}"
        self.fixed = fixed
        self.rot = False
        self.x = 0
        self.y = 0

    @property
    def w(self):
        return self.h0 if self.rot else self.w0

    @property
    def h(self):
        return self.w0 if self.rot else self.h0

    def rect_cells(self):
        for xi in range(self.x, self.x + self.w):
            for yi in range(self.y, self.y + self.h):
                yield (xi, yi)

    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def compute_blocked_cells(rooms, plot_w, plot_h):
    blocked = set()
    for r in rooms:
        for x, y in r.rect_cells():
            if 0 <= x < plot_w and 0 <= y < plot_h:
                blocked.add((x, y))
    return blocked


def room_perimeter_targets(r, plot_w, plot_h):
    targets = set()
    for xi in range(r.x, r.x + r.w):
        targets.add((xi, r.y - 1))
        targets.add((xi, r.y + r.h))
    for yi in range(r.y, r.y + r.h):
        targets.add((r.x - 1, yi))
        targets.add((r.x + r.w, yi))
    return set((x, y) for (x, y) in targets if 0 <= x < plot_w and 0 <= y < plot_h)


def choose_seed_cell(free_cells, plot_w, plot_h):
    if not free_cells:
        return None
    cx = plot_w // 2
    cy = plot_h // 2
    best = None
    bestd = None
    for c in free_cells:
        d = abs(c[0] - cx) + abs(c[1] - cy)
        if best is None or d < bestd:
            best = c
            bestd = d
    return best


def build_backbone_guided(rooms, plot_w, plot_h, blocked, seed=None, max_expand=None):
    free = set(
        (x, y) for x in range(plot_w) for y in range(plot_h) if (x, y) not in blocked
    )
    if not free:
        return set(), set(), False
    start = seed if seed and seed in free else choose_seed_cell(free, plot_w, plot_h)
    if start is None:
        start = next(iter(free))
    # compute multi-source distance from room perimeters
    perim_sources = set()
    for r in rooms:
        perim_sources |= room_perimeter_targets(r, plot_w, plot_h)
    perim_sources = set(p for p in perim_sources if p in free)
    dist = {}
    dq = deque()
    for p in perim_sources:
        dist[p] = 0
        dq.append(p)
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            nb = (nx, ny)
            if 0 <= nx < plot_w and 0 <= ny < plot_h and nb in free and nb not in dist:
                dist[nb] = dist[(x, y)] + 1
                dq.append(nb)
    from heapq import heappush, heappop

    heap = []
    visited = set([start])
    start_priority = dist.get(start, 10000)
    heappush(heap, (start_priority, 0, start))
    backbone = set([start])
    touched = set()
    steps = 0
    limit = max_expand if max_expand is not None else len(free)
    while heap and steps < limit and len(touched) < len(rooms):
        priority, _, cur = heappop(heap)
        x, y = cur
        # check touch
        for r in rooms:
            if r.idx in touched:
                continue
            if cur in room_perimeter_targets(r, plot_w, plot_h):
                touched.add(r.idx)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            nb = (nx, ny)
            if not (0 <= nx < plot_w and 0 <= ny < plot_h):
                continue
            if nb in visited:
                continue
            if nb in blocked:
                continue
            visited.add(nb)
            backbone.add(nb)
            p = dist.get(nb, 10000)
            heappush(heap, (p, steps, nb))
        steps += 1
    finished = len(touched) == len(rooms)
    return backbone, touched, finished


def prune_and_reconnect_corridor(
    backbone, rooms, plot_w, plot_h, blocked, prune_radius=1
):
    perim_all = set()
    room_perims = {}
    for r in rooms:
        p = room_perimeter_targets(r, plot_w, plot_h)
        room_perims[r.idx] = p
        perim_all |= p
    adjacency = {}
    for c in backbone:
        x, y = c
        cnt = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) in perim_all:
                cnt += 1
        adjacency[c] = cnt
    hot = [c for c, v in adjacency.items() if v > 0]
    if not hot:
        keep = set(backbone)
    else:
        dq = deque()
        seen = set()
        keep = set()
        for c in hot:
            dq.append((c, 0))
            seen.add(c)
            keep.add(c)
        while dq:
            c, d = dq.popleft()
            if d >= prune_radius:
                continue
            x, y = c
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in backbone and nb not in seen:
                    seen.add(nb)
                    keep.add(nb)
                    dq.append((nb, d + 1))

    def components(cset):
        comps = []
        visited = set()
        for cell in cset:
            if cell in visited:
                continue
            dq = deque([cell])
            comp = set([cell])
            visited.add(cell)
            while dq:
                cx, cy = dq.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (cx + dx, cy + dy)
                    if nb in cset and nb not in visited:
                        visited.add(nb)
                        comp.add(nb)
                        dq.append(nb)
            comps.append(comp)
        return comps

    comps = components(keep)
    final = set(keep)
    if len(comps) > 1:
        comps_sorted = sorted(comps, key=lambda c: -len(c))
        main = comps_sorted[0]
        others = comps_sorted[1:]
        for comp in others:
            best = None
            bestd = 10**9
            for a in main:
                for b in comp:
                    d = abs(a[0] - b[0]) + abs(a[1] - b[1])
                    if d < bestd:
                        bestd = d
                        best = (a, b)
            if best is None:
                continue
            a, b = best
            (bx, by), (ax, ay) = b, a
            elbow1 = (bx, ay)
            elbow2 = (ax, by)
            chosen = None
            for elbow in (elbow1, elbow2):
                good = True
                segs = [(b, elbow), (elbow, a)]
                for p, q in segs:
                    x1, y1 = p
                    x2, y2 = q
                    if x1 == x2:
                        for y in range(min(y1, y2), max(y1, y2) + 1):
                            if (x1, y) in blocked:
                                good = False
                                break
                    else:
                        for x in range(min(x1, x2), max(x1, x2) + 1):
                            if (x, y1) in blocked:
                                good = False
                                break
                    if not good:
                        break
                if good:
                    chosen = elbow
                    break
            if chosen is None:
                chosen = elbow1

            def add_seg(p, q):
                (x1, y1), (x2, y2) = p, q
                if x1 == x2:
                    for y in range(min(y1, y2), max(y1, y2) + 1):
                        if (x1, y) not in blocked:
                            final.add((x1, y))
                else:
                    for x in range(min(x1, x2), max(x1, x2) + 1):
                        if (x, y1) not in blocked:
                            final.add((x, y1))

            add_seg(b, chosen)
            add_seg(chosen, a)
            main = main.union(comp).union(set([chosen]))
    return final


def dilate_mask(mask, blocked, plot_w, plot_h, target_width):
    if target_width <= 1:
        return set(mask)
    radius = max(0, (target_width - 1) // 2)
    cur = set(mask)
    for _ in range(radius):
        new = set(cur)
        for x, y in cur:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < plot_w and 0 <= ny < plot_h and (nx, ny) not in blocked:
                    new.add((nx, ny))
        cur = new
    return cur


def compute_blocked_for_rooms(rooms, plot_w, plot_h, ignore_room_idx=None):
    blocked = set()
    for r in rooms:
        if ignore_room_idx is not None and r.idx == ignore_room_idx:
            continue
        for c in r.rect_cells():
            if 0 <= c[0] < plot_w and 0 <= c[1] < plot_h:
                blocked.add(c)
    return blocked


def try_nudge_room_once(room, rooms, plot_w, plot_h, blocked, backbone, max_step=1):
    best = None
    best_gain = -1
    for dx in range(-max_step, max_step + 1):
        for dy in range(-max_step, max_step + 1):
            if dx == 0 and dy == 0:
                continue
            nx = room.x + dx
            ny = room.y + dy
            if nx < 0 or ny < 0 or nx + room.w > plot_w or ny + room.h > plot_h:
                continue
            collide = False
            for r2 in rooms:
                if r2.idx == room.idx:
                    continue
                if not (
                    nx + room.w - 1 < r2.x
                    or nx > r2.x + r2.w - 1
                    or ny + room.h - 1 < r2.y
                    or ny > r2.y + r2.h - 1
                ):
                    collide = True
                    break
            if collide:
                continue
            perim = set()
            for xi in range(nx, nx + room.w):
                perim.add((xi, ny - 1))
                perim.add((xi, ny + room.h))
            for yi in range(ny, ny + room.h):
                perim.add((nx - 1, yi))
                perim.add((nx + room.w, yi))
            valid_perim = set(
                p
                for p in perim
                if 0 <= p[0] < plot_w and 0 <= p[1] < plot_h and p not in blocked
            )
            new_adj = sum(1 for p in valid_perim if p in backbone)
            new_perim_cells = len(valid_perim)
            gain = new_adj * 25 + new_perim_cells
            if gain > best_gain:
                best_gain = gain
                best = (dx, dy, gain)
    if best is None:
        return 0, 0, False, 0
    dx, dy, gain = best
    return dx, dy, gain > 0, gain


def repair_by_nudges(
    rooms,
    plot_w,
    plot_h,
    blocked,
    initial_backbone,
    max_nudges_total=8,
    per_room_max_step=1,
):
    rooms_copy = copy.deepcopy(rooms)
    blocked_set = compute_blocked_cells(rooms_copy, plot_w, plot_h)
    seed = choose_seed_cell(
        set(
            (x, y)
            for x in range(plot_w)
            for y in range(plot_h)
            if (x, y) not in blocked_set
        ),
        plot_w,
        plot_h,
    )
    backbone, touched, finished = build_backbone_guided(
        rooms_copy, plot_w, plot_h, blocked_set, seed=seed
    )
    nudges_used = 0
    while len(touched) < len(rooms_copy) and nudges_used < max_nudges_total:
        blocking = [r for r in rooms_copy if r.idx not in touched]
        best_score = -1
        best_action = None
        for r in blocking:
            dx, dy, improved, score = try_nudge_room_once(
                r,
                rooms_copy,
                plot_w,
                plot_h,
                compute_blocked_for_rooms(
                    rooms_copy, plot_w, plot_h, ignore_room_idx=r.idx
                ),
                backbone,
                max_step=per_room_max_step,
            )
            if improved and score > best_score:
                best_score = score
                best_action = (r.idx, dx, dy, score)
        if best_action is None:
            for r in blocking:
                dx, dy, improved, score = try_nudge_room_once(
                    r,
                    rooms_copy,
                    plot_w,
                    plot_h,
                    compute_blocked_for_rooms(
                        rooms_copy, plot_w, plot_h, ignore_room_idx=r.idx
                    ),
                    backbone,
                    max_step=2,
                )
                if improved and score > best_score:
                    best_score = score
                    best_action = (r.idx, dx, dy, score)
        if best_action is None:
            break
        rid, dx, dy, score = best_action
        rc = next(r for r in rooms_copy if r.idx == rid)
        rc.x += dx
        rc.y += dy
        nudges_used += 1
        blocked_set = compute_blocked_cells(rooms_copy, plot_w, plot_h)
        seed = choose_seed_cell(
            set(
                (x, y)
                for x in range(plot_w)
                for y in range(plot_h)
                if (x, y) not in blocked_set
            ),
            plot_w,
            plot_h,
        )
        backbone, touched, finished = build_backbone_guided(
            rooms_copy, plot_w, plot_h, blocked_set, seed=seed
        )
    success = len(touched) == len(rooms_copy)
    return rooms_copy, nudges_used, backbone, touched, success


# ----------------------------
# Energy: tuned constants (adjacency reward large, remote_penalty large)
# ----------------------------
def energy_of_layout(
    rooms,
    plot_w,
    plot_h,
    backbone,
    nudges_used,
    plot_area_max_rooms,
    adjacency_reward=1200.0,
    remote_penalty=50.0,
):
    overlap = 0
    oob = 0
    grid = {}
    for r in rooms:
        for x, y in r.rect_cells():
            if not (0 <= x < plot_w and 0 <= y < plot_h):
                oob += 1
                continue
            if (x, y) in grid:
                overlap += 1
            else:
                grid[(x, y)] = r.idx
    if overlap > 0 or oob > 0:
        return 1e9 + overlap * 1e6 + oob * 1e6
    minx = min(r.x for r in rooms)
    miny = min(r.y for r in rooms)
    maxx = max(r.x + r.w - 1 for r in rooms)
    maxy = max(r.y + r.h - 1 for r in rooms)
    bbox_area = (maxx - minx + 1) * (maxy - miny + 1)
    spanned = 0
    perim_all = set()
    for r in rooms:
        per = room_perimeter_targets(r, plot_w, plot_h)
        perim_all |= per
        if per & backbone:
            spanned += 1
    remote_cells = 0
    for c in backbone:
        x, y = c
        adj = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) in perim_all:
                adj += 1
        if adj == 0:
            remote_cells += 1
    corridor_len = len(backbone)
    total_rooms_area = sum(r.w * r.h for r in rooms)
    penalty_area = 0
    if total_rooms_area > plot_area_max_rooms:
        penalty_area = (total_rooms_area - plot_area_max_rooms) * 10000
    # tuned energy: minimize bbox, maximize adjacency, penalize remote corridor, penalize nudges lightly
    E = (
        1.0 * bbox_area
        - adjacency_reward * spanned
        + 1.2 * corridor_len
        + remote_penalty * remote_cells
        + 2200.0 * nudges_used
        + penalty_area
    )
    return E


# ----------------------------
# Neighbor generator: stronger center pull
# ----------------------------
def neighbor_rooms_state(rooms, plot_w, plot_h, center_pull_prob=0.25):
    ns = copy.deepcopy(rooms)
    r = random.random()
    if r < 0.55:
        room = random.choice(ns)
        if room.fixed:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
            room.x = clamp(room.x + dx, 0, max(0, plot_w - room.w))
            room.y = clamp(room.y + dy, 0, max(0, plot_h - room.h))
        else:
            if random.random() < 0.18:
                room.rot = not room.rot
                room.x = clamp(room.x, 0, max(0, plot_w - room.w))
                room.y = clamp(room.y, 0, max(0, plot_h - room.h))
            else:
                if random.random() < center_pull_prob:
                    cx = plot_w / 2.0
                    cy = plot_h / 2.0
                    rx, ry = room.center()
                    dx = int(math.copysign(1, cx - rx)) if abs(cx - rx) >= 1 else 0
                    dy = int(math.copysign(1, cy - ry)) if abs(cy - ry) >= 1 else 0
                    if random.random() < 0.3:
                        dx *= random.randint(1, 3)
                        dy *= random.randint(1, 3)
                    room.x = clamp(room.x + dx, 0, max(0, plot_w - room.w))
                    room.y = clamp(room.y + dy, 0, max(0, plot_h - room.h))
                else:
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    room.x = clamp(room.x + dx, 0, max(0, plot_w - room.w))
                    room.y = clamp(room.y + dy, 0, max(0, plot_h - room.h))
    elif r < 0.8:
        a, b = random.sample(ns, 2)
        if not (a.fixed or b.fixed):
            a.x, b.x = b.x, a.x
            a.y, b.y = b.y, a.y
            a.rot, b.rot = b.rot, a.rot
    else:
        room = random.choice(ns)
        if not room.fixed:
            room.rot = random.choice([False, True])
            room.x = random.randint(0, max(0, plot_w - room.w))
            room.y = random.randint(0, max(0, plot_h - room.h))
    return ns


# ----------------------------
# Post-refinement: local sweeps (greedy)
# ----------------------------
def post_refine(rooms, plot_w, plot_h, backbone, corridor_width, plot_area_max_rooms):
    # attempt small local moves that reduce energy; few sweeps
    best_rooms = copy.deepcopy(rooms)
    best_backbone = set(backbone)
    best_E = energy_of_layout(
        best_rooms, plot_w, plot_h, best_backbone, 0, plot_area_max_rooms
    )
    for sweep in range(8):
        improved = False
        for r in best_rooms:
            # try moves in small neighborhood
            original = (r.x, r.y, r.rot)
            neighbors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = clamp(r.x + dx, 0, max(0, plot_w - r.w))
                    ny = clamp(r.y + dy, 0, max(0, plot_h - r.h))
                    neighbors.append((nx, ny, r.rot))
            for nx, ny, nrot in neighbors:
                r.x, r.y, r.rot = nx, ny, nrot
                blocked = compute_blocked_cells(best_rooms, plot_w, plot_h)
                seed = choose_seed_cell(
                    set(
                        (x, y)
                        for x in range(plot_w)
                        for y in range(plot_h)
                        if (x, y) not in blocked
                    ),
                    plot_w,
                    plot_h,
                )
                backbone_cand, touched, finished = build_backbone_guided(
                    best_rooms, plot_w, plot_h, blocked, seed=seed
                )
                pr = prune_and_reconnect_corridor(
                    backbone_cand, best_rooms, plot_w, plot_h, blocked, prune_radius=1
                )
                backbone_cand = dilate_mask(pr, blocked, plot_w, plot_h, corridor_width)
                E = energy_of_layout(
                    best_rooms,
                    plot_w,
                    plot_h,
                    backbone_cand,
                    0,
                    plot_area_max_rooms,
                    adjacency_reward=1200.0,
                    remote_penalty=50.0,
                )
                if E < best_E:
                    best_E = E
                    best_backbone = set(backbone_cand)
                    improved = True
                    break
                else:
                    r.x, r.y, r.rot = original
            # end neighbor loop
        if not improved:
            break
    return best_rooms, best_backbone, best_E


# ----------------------------
# SA driver with multi-restarts and improved tuning
# ----------------------------
def run_sa_with_restarts(
    plot_w,
    plot_h,
    rooms_input,
    corridor_width=3,
    T0=2000.0,
    alpha=0.995,
    iter_per_temp=200,
    max_total_iters=4000,
    strict_mode=False,
    nudges_budget=8,
    nudges_per_room_step=1,
    plot_area_fraction=0.7,
    restarts=3,
    progress_cb=None,
    center_pull_prob=0.25,
):
    best_overall = None
    for restart in range(restarts):
        # run one SA (similar to previous improved algorithm)
        rooms = copy.deepcopy(rooms_input)
        rs = sorted(rooms, key=lambda r: r.w0 * r.h0, reverse=True)
        occ = [[False] * plot_h for _ in range(plot_w)]
        for r in rs:
            r.rot = random.choice([False, True])
            placed = False
            for y in range(0, plot_h - r.h + 1):
                for x in range(0, plot_w - r.w + 1):
                    ok = True
                    for xi in range(x, x + r.w):
                        for yi in range(y, y + r.h):
                            if occ[xi][yi]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        r.x = x
                        r.y = y
                        for xi in range(x, x + r.w):
                            for yi in range(y, y + r.h):
                                occ[xi][yi] = True
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                r.x = random.randint(0, max(0, plot_w - r.w))
                r.y = random.randint(0, max(0, plot_h - r.h))
        plot_area_max_rooms = plot_area_fraction * (plot_w * plot_h)
        blocked = compute_blocked_cells(rooms, plot_w, plot_h)
        seed = choose_seed_cell(
            set(
                (x, y)
                for x in range(plot_w)
                for y in range(plot_h)
                if (x, y) not in blocked
            ),
            plot_w,
            plot_h,
        )
        backbone, touched, finished = build_backbone_guided(
            rooms, plot_w, plot_h, blocked, seed=seed
        )
        nudges_used = 0
        rooms_after = rooms
        success = finished
        if not finished and not strict_mode:
            rooms_after, nudges_used, backbone, touched, success = repair_by_nudges(
                rooms,
                plot_w,
                plot_h,
                blocked,
                backbone,
                max_nudges_total=nudges_budget,
                per_room_max_step=nudges_per_room_step,
            )
            blocked = compute_blocked_cells(rooms_after, plot_w, plot_h)
        pruned = prune_and_reconnect_corridor(
            backbone, rooms_after, plot_w, plot_h, blocked, prune_radius=1
        )
        backbone = dilate_mask(
            pruned,
            compute_blocked_cells(rooms_after, plot_w, plot_h),
            plot_w,
            plot_h,
            corridor_width,
        )
        best_rooms = copy.deepcopy(rooms_after)
        best_backbone = set(backbone)
        best_nudges = nudges_used
        best_energy = energy_of_layout(
            best_rooms,
            plot_w,
            plot_h,
            best_backbone,
            best_nudges,
            plot_area_max_rooms,
            adjacency_reward=1200.0,
            remote_penalty=50.0,
        )
        # SA loop (iterations)
        T = T0
        total_iters = 0
        while total_iters < max_total_iters:
            for _ in range(iter_per_temp):
                if total_iters >= max_total_iters:
                    break
                cand_rooms = neighbor_rooms_state(
                    rooms_after, plot_w, plot_h, center_pull_prob=center_pull_prob
                )
                blocked_cand = compute_blocked_cells(cand_rooms, plot_w, plot_h)
                seed_c = choose_seed_cell(
                    set(
                        (x, y)
                        for x in range(plot_w)
                        for y in range(plot_h)
                        if (x, y) not in blocked_cand
                    ),
                    plot_w,
                    plot_h,
                )
                backbone_cand, touched_cand, finished_cand = build_backbone_guided(
                    cand_rooms,
                    plot_w,
                    plot_h,
                    blocked_cand,
                    seed=seed_c,
                    max_expand=plot_w * plot_h // 3,
                )
                nudges_cand = 0
                cand_rooms_after = cand_rooms
                cand_success = finished_cand
                if not finished_cand:
                    if strict_mode:
                        cand_E = 1e9
                    else:
                        (
                            cand_rooms_after,
                            nudges_cand,
                            backbone_cand,
                            touched_cand,
                            cand_success,
                        ) = repair_by_nudges(
                            cand_rooms,
                            plot_w,
                            plot_h,
                            blocked_cand,
                            backbone_cand,
                            max_nudges_total=min(nudges_budget, 3),
                            per_room_max_step=nudges_per_room_step,
                        )
                if cand_success:
                    pr = prune_and_reconnect_corridor(
                        backbone_cand,
                        cand_rooms_after,
                        plot_w,
                        plot_h,
                        compute_blocked_cells(cand_rooms_after, plot_w, plot_h),
                        prune_radius=1,
                    )
                    backbone_cand = dilate_mask(
                        pr,
                        compute_blocked_cells(cand_rooms_after, plot_w, plot_h),
                        plot_w,
                        plot_h,
                        corridor_width,
                    )
                if not cand_success and strict_mode:
                    cand_E = 1e9
                else:
                    cand_E = energy_of_layout(
                        cand_rooms_after,
                        plot_w,
                        plot_h,
                        backbone_cand,
                        nudges_cand,
                        plot_area_max_rooms,
                        adjacency_reward=1200.0,
                        remote_penalty=50.0,
                    )
                cur_E = energy_of_layout(
                    rooms_after,
                    plot_w,
                    plot_h,
                    backbone,
                    nudges_used,
                    plot_area_max_rooms,
                    adjacency_reward=1200.0,
                    remote_penalty=50.0,
                )
                dE = cand_E - cur_E
                accept = False
                if dE <= 0 or random.random() < math.exp(-dE / max(1e-9, T)):
                    accept = True
                    rooms_after = cand_rooms_after
                    backbone = backbone_cand
                    nudges_used = nudges_cand
                    success = cand_success
                if accept:
                    if cand_E < best_energy:
                        best_energy = cand_E
                        best_rooms = copy.deepcopy(rooms_after)
                        best_backbone = set(backbone)
                        best_nudges = nudges_used
                total_iters += 1
            T *= alpha
        # post-refine the best from this restart
        refined_rooms, refined_backbone, refined_E = post_refine(
            best_rooms,
            plot_w,
            plot_h,
            best_backbone,
            corridor_width,
            plot_area_max_rooms,
        )
        if best_overall is None or refined_E < best_overall["energy"]:
            best_overall = {
                "rooms": refined_rooms,
                "backbone": refined_backbone,
                "energy": refined_E,
                "nudges": best_nudges,
                "success": True,
            }
        if progress_cb:
            progress_cb(
                restart + 1, restarts, best_overall["energy"] if best_overall else None
            )
    return best_overall


# ----------------------------
# GUI
# ----------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("SA Compact Plaza (tuned)")
        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)
        ctrl = ttk.Frame(self.frame, padding=6)
        ctrl.pack(side="left", fill="y")

        ttk.Label(ctrl, text="Plot width").grid(row=0, column=0)
        self.plot_w_var = tk.IntVar(value=32)
        ttk.Entry(ctrl, textvariable=self.plot_w_var, width=8).grid(row=0, column=1)
        ttk.Label(ctrl, text="Plot height").grid(row=1, column=0)
        self.plot_h_var = tk.IntVar(value=36)
        ttk.Entry(ctrl, textvariable=self.plot_h_var, width=8).grid(row=1, column=1)

        ttk.Label(ctrl, text="Rooms (w,h[,name]) per line").grid(
            row=2, column=0, columnspan=2
        )
        self.rooms_txt = tk.Text(ctrl, width=36, height=12)
        self.rooms_txt.grid(row=3, column=0, columnspan=2)
        self.rooms_txt.insert(
            "1.0",
            """8,6,Meeting room
15,10,Conf room
9,9,Meeting room
10,12,MD Cabin1
10,12,MD Cabin2
7,6,Cabin2
7,6,Cabin3
8,6,Workstations1
8,6,Workstations2
8,6,Workstations3
""",
        )

        ttk.Label(ctrl, text="Corridor width (>=3)").grid(row=4, column=0)
        self.cw_var = tk.IntVar(value=5)
        ttk.Entry(ctrl, textvariable=self.cw_var, width=8).grid(row=4, column=1)

        ttk.Label(ctrl, text="Total iters per restart").grid(row=5, column=0)
        self.iter_var = tk.IntVar(value=4000)
        ttk.Entry(ctrl, textvariable=self.iter_var, width=8).grid(row=5, column=1)
        ttk.Label(ctrl, text="Restarts (per layout)").grid(row=6, column=0)
        self.restarts_var = tk.IntVar(value=3)
        ttk.Entry(ctrl, textvariable=self.restarts_var, width=8).grid(row=6, column=1)

        ttk.Label(ctrl, text="Nudges budget").grid(row=7, column=0)
        self.nudge_var = tk.IntVar(value=8)
        ttk.Entry(ctrl, textvariable=self.nudge_var, width=8).grid(row=7, column=1)

        ttk.Label(ctrl, text="Num layouts").grid(row=8, column=0)
        self.num_var = tk.IntVar(value=3)
        ttk.Entry(ctrl, textvariable=self.num_var, width=8).grid(row=8, column=1)

        # sliders for tuning knobs
        ttk.Label(ctrl, text="Adjacency reward").grid(row=9, column=0)
        self.adj_s = tk.DoubleVar(value=1200.0)
        ttk.Scale(
            ctrl, variable=self.adj_s, from_=200, to=2000, orient="horizontal"
        ).grid(row=9, column=1)
        ttk.Label(ctrl, text="Remote penalty").grid(row=10, column=0)
        self.rem_s = tk.DoubleVar(value=50.0)
        ttk.Scale(ctrl, variable=self.rem_s, from_=0, to=200, orient="horizontal").grid(
            row=10, column=1
        )
        ttk.Label(ctrl, text="Center pull prob").grid(row=11, column=0)
        self.cp_s = tk.DoubleVar(value=0.25)
        ttk.Scale(
            ctrl, variable=self.cp_s, from_=0.0, to=0.6, orient="horizontal"
        ).grid(row=11, column=1)

        self.strict_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Strict (no nudges)", variable=self.strict_var).grid(
            row=12, column=0, columnspan=2
        )

        self.generate_btn = ttk.Button(
            ctrl, text="Generate", command=self.generate_layouts
        )
        self.generate_btn.grid(row=13, column=0, columnspan=2, pady=(6, 0))
        self.refine_btn = ttk.Button(
            ctrl, text="Refine current", command=self.refine_current, state="disabled"
        )
        self.refine_btn.grid(row=14, column=0, columnspan=2, pady=(6, 0))

        nav = ttk.Frame(ctrl)
        nav.grid(row=15, column=0, columnspan=2, pady=(6, 0))
        self.prev_btn = ttk.Button(
            nav, text="◀ Prev", command=self.prev_layout, state="disabled"
        )
        self.prev_btn.pack(side="left", padx=6)
        self.next_btn = ttk.Button(
            nav, text="Next ▶", command=self.next_layout, state="disabled"
        )
        self.next_btn.pack(side="right", padx=6)

        canvas_fr = ttk.Frame(self.frame)
        canvas_fr.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_fr, width=1000, height=900, bg="white")
        self.canvas.pack(fill="both", expand=True)

        self.layouts = []
        self.current = 0
        self.worker = None

    def parse_rooms(self):
        s = self.rooms_txt.get("1.0", "end").strip()
        rooms = []
        idx = 0
        for ln in s.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(",")]
            try:
                w = int(parts[0])
                h = int(parts[1])
                name = parts[2] if len(parts) > 2 else None
                rooms.append(Room(w, h, idx, name=name))
                idx += 1
            except:
                continue
        return rooms

    def generate_layouts(self):
        rooms = self.parse_rooms()
        if not rooms:
            messagebox.showerror("Error", "No rooms parsed")
            return
        plot_w = self.plot_w_var.get()
        plot_h = self.plot_h_var.get()
        total_area = sum(r.w0 * r.h0 for r in rooms)
        if total_area > 0.7 * plot_w * plot_h:
            if not messagebox.askyesno(
                "Area >70%", "Total rooms area exceeds 70% of plot. Continue?"
            ):
                return
        cw = max(3, int(self.cw_var.get()))
        num = max(1, int(self.num_var.get()))
        restarts = max(1, int(self.restarts_var.get()))
        self.generate_btn.config(state="disabled")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")
        self.refine_btn.config(state="disabled")
        self.canvas.delete("status")
        self.canvas.create_text(
            200, 200, text="Running SA (tuned)...", tags=("status",), fill="black"
        )
        t = threading.Thread(
            target=self._worker,
            args=(plot_w, plot_h, rooms, cw, num, restarts),
            daemon=True,
        )
        t.start()
        self.worker = t

    def _worker(self, plot_w, plot_h, rooms, cw, num, restarts):
        outs = []
        for i in range(num):
            res = run_sa_with_restarts(
                plot_w,
                plot_h,
                rooms,
                corridor_width=cw,
                T0=2000.0,
                alpha=0.995,
                iter_per_temp=200,
                max_total_iters=self.iter_var.get(),
                strict_mode=self.strict_var.get(),
                nudges_budget=self.nudge_var.get(),
                nudges_per_room_step=1,
                plot_area_fraction=0.7,
                restarts=restarts,
                progress_cb=None,
                center_pull_prob=self.cp_s.get(),
            )
            if res:
                res["energy"] = energy_of_layout(
                    res["rooms"],
                    plot_w,
                    plot_h,
                    res["backbone"],
                    res.get("nudges", 0),
                    0.7 * (plot_w * plot_h),
                    adjacency_reward=self.adj_s.get(),
                    remote_penalty=self.rem_s.get(),
                )
            outs.append(res)
        self.root.after(0, lambda: self._done(outs))

    def _done(self, layouts):
        self.layouts = layouts
        self.current = 0
        self.generate_btn.config(state="normal")
        if self.layouts:
            self.prev_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.refine_btn.config(state="normal")
            self.draw()
            messagebox.showinfo("Done", f"Generated {len(self.layouts)} layouts")
        self.canvas.delete("status")

    def prev_layout(self):
        if not self.layouts:
            return
        self.current = (self.current - 1) % len(self.layouts)
        self.draw()

    def next_layout(self):
        if not self.layouts:
            return
        self.current = (self.current + 1) % len(self.layouts)
        self.draw()

    def refine_current(self):
        if not self.layouts:
            return
        st = self.layouts[self.current]
        plot_w = self.plot_w_var.get()
        plot_h = self.plot_h_var.get()
        cw = max(3, int(self.cw_var.get()))
        plot_area_max_rooms = 0.7 * (plot_w * plot_h)
        refined_rooms, refined_backbone, refined_E = post_refine(
            st["rooms"], plot_w, plot_h, st["backbone"], cw, plot_area_max_rooms
        )
        st["rooms"] = refined_rooms
        st["backbone"] = refined_backbone
        st["energy"] = refined_E
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        if not self.layouts:
            return
        st = self.layouts[self.current]
        rooms = st["rooms"]
        backbone = st["backbone"]
        plot_w = self.plot_w_var.get()
        plot_h = self.plot_h_var.get()
        W = max(800, int(self.canvas.winfo_width()))
        H = max(600, int(self.canvas.winfo_height()))
        margin = 24
        sx = (W - 2 * margin) / plot_w
        sy = (H - 2 * margin) / plot_h
        s = min(sx, sy)
        ox = margin
        oy = margin

        def to_canvas(x, y):
            return ox + x * s, oy + (plot_h - (y + 1)) * s

        gc = "#e8e8e8"
        for gx in range(plot_w + 1):
            x1, y1 = to_canvas(gx, 0)
            x2, y2 = to_canvas(gx, plot_h)
            self.canvas.create_line(x1, y1, x2, y2, fill=gc)
        for gy in range(plot_h + 1):
            x1, y1 = to_canvas(0, gy)
            x2, y2 = to_canvas(plot_w, gy)
            self.canvas.create_line(x1, y1, x2, y2, fill=gc)
        for cx, cy in backbone:
            x1, y1 = to_canvas(cx, cy)
            x2, y2 = to_canvas(cx + 1, cy + 1)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#C0C0C0", outline="")
        for r in rooms:
            x1, y1 = to_canvas(r.x, r.y)
            x2, y2 = to_canvas(r.x + r.w, r.y + r.h)
            random.seed(r.idx + 33)
            col = "#%02x%02x%02x" % (
                random.randint(80, 200),
                random.randint(80, 200),
                random.randint(80, 200),
            )
            self.canvas.create_rectangle(x1, y2, x2, y1, fill=col, outline="black")
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=f"{r.name}\n{r.w}x{r.h}",
                fill="white",
            )
        energy_str = f"Energy={st['energy']:.1f} Nudges={st.get('nudges',0)} Layout {self.current+1}/{len(self.layouts)}"
        self.canvas.create_text(8, 8, anchor="nw", text=energy_str, fill="black")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
