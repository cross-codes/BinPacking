import tkinter as tk
from tkinter import ttk, messagebox
from struct.Layout import Layout
from constants import *
from PackingHeuristics import PackingHeuristics
from struct.Corridor import Orientation


class App:
    def __init__(self, root: tk.Tk):
        self.root: tk.Tk = root
        self.valid_layouts: list[Layout] = []
        self.idx: int = 0
        self.plot_w_var: tk.StringVar = tk.StringVar(value="40")
        self.plot_h_var: tk.StringVar = tk.StringVar(value="25")
        self.status_var: tk.StringVar = tk.StringVar(
            value='Enter plot size and rooms, then click "Generate valid layouts"'
        )

        root.title("Planner")

        ctrl = ttk.Frame(root)
        # Multi-corridor controls
        self.use_multi_var = tk.BooleanVar(value=True)
        self.corridor_k_var = tk.StringVar(value="")  # empty = auto

        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(ctrl, text="Total plot width").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.plot_w_var, width=2).grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Label(ctrl, text="Total plot height").grid(
            row=0, column=2, sticky="w", padx=(16, 0)
        )
        ttk.Entry(ctrl, textvariable=self.plot_h_var, width=2).grid(
            row=0, column=3, sticky="w", padx=4
        )
        ttk.Button(ctrl, text="Generate valid layouts", command=self.generate).grid(
            row=0, column=4, padx=(16, 4)
        )
        ttk.Button(ctrl, text="Previous layout", command=self.prev).grid(
            row=0, column=5, padx=2
        )
        ttk.Button(ctrl, text="Next layout", command=self.next).grid(
            row=0, column=6, padx=2
        )
        ttk.Button(ctrl, text="Load test sample", command=self.on_sample).grid(
            row=0, column=7, padx=(16, 2)
        )
        self.allow_multi_var = tk.BooleanVar(value=True)   # single source of truth
        ttk.Checkbutton(
            ctrl,
            text="Try atmost corridors",
            variable=self.allow_multi_var,
            command=lambda: self.k_spinbox.config(
                state=("normal" if self.allow_multi_var.get() else "disabled")
            ),
        ).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))

        ttk.Label(ctrl, text="Corridors (k)").grid(row=1, column=1, sticky="e", pady=(6, 0))
        self.k_var = tk.StringVar(value="")  # empty = auto (maximize)
        self.k_spinbox = tk.Spinbox(ctrl, from_=1, to=99, textvariable=self.k_var, width=5, state="normal")
        self.k_spinbox.grid(row=1, column=2, sticky="w", padx=(4, 12), pady=(6, 0))

        ttk.Label(ctrl, text="Show").grid(row=1, column=3, sticky="e", padx=(0, 4), pady=(6, 0))
        self.heuristic_filter_var = tk.StringVar(value="All")
        self.heuristic_filter = ttk.Combobox(
            ctrl,
            textvariable=self.heuristic_filter_var,
            state="readonly",
            width=16,
            values=["All", "Multi-vertical", "Single-vertical", "Single-horizontal"],
        )
        self.heuristic_filter.grid(row=1, column=4, sticky="w", pady=(6, 0))


        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        rooms_frame = ttk.LabelFrame(
            main_frame,
            text="Rooms (one per line: <name> <width> <height> OR <width> <height>)",
        )
        rooms_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.rooms_text: tk.Text = tk.Text(rooms_frame, width=36, height=25)
        self.rooms_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.rooms_text.insert(
            "1.0",
            "R1 6 5\nR2 7 4\nR3 4 4\nR4 5 5\nR5 3 8\nR6 6 6\nR7 9 3\nR8 4 7\nR9 3 3\nR10 4.5 4\n",
        )

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas: tk.Canvas = tk.Canvas(
            right_frame, width=CANVAS_W, height=CANVAS_H, bg="white"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, textvariable=self.status_var).pack(
            side=tk.BOTTOM, anchor="w"
        )

    def generate(self):
        try:
            plot_w = float(self.plot_w_var.get())
            plot_h = float(self.plot_h_var.get())
        except ValueError:
            _ = messagebox.showerror(
                "Invalid input", "Plot width/height must be numbers."
            )
            return

        try:
            rooms = PackingHeuristics.parse_rooms(self.rooms_text.get("1.0", tk.END))
        except Exception as e:
            _ = messagebox.showerror("Invalid rooms", str(e))
            return

        if not rooms:
            _ = messagebox.showinfo("No rooms", "Please enter at least one room.")
            return

        # Decide desired_k based on checkbox
        # Decide desired_k based on checkbox
        desired_k = None
        if self.allow_multi_var.get():
            k_text = self.k_var.get().strip()
            desired_k = int(k_text) if k_text.isdigit() and int(k_text) >= 1 else None

        self.valid_layouts = PackingHeuristics.generate_layouts(plot_w, plot_h, rooms, desired_k=desired_k)

        # Filter by heuristic selection
        f = self.heuristic_filter_var.get()
        if not self.allow_multi_var.get():
            self.valid_layouts = [L for L in self.valid_layouts if getattr(L, "heuristic", "") != "multi-vertical"]
        if f == "Multi-vertical":
            self.valid_layouts = [L for L in self.valid_layouts if getattr(L, "heuristic", "") == "multi-vertical"]
        elif f == "Single-vertical":
            self.valid_layouts = [L for L in self.valid_layouts if getattr(L, "heuristic", "") == "single-vertical"]
        elif f == "Single-horizontal":
            self.valid_layouts = [L for L in self.valid_layouts if getattr(L, "heuristic", "") == "single-horizontal"]



        if not self.allow_multi_var.get():
            self.valid_layouts = [
                L for L in self.valid_layouts
                if getattr(L, "heuristic", "") != "multi-vertical"
            ]

        f = self.heuristic_filter_var.get()
        if f == "Multi-vertical":
            self.valid_layouts = [
                L for L in self.valid_layouts
                if getattr(L, "heuristic", "") == "multi-vertical"
            ]
        elif f == "Single-vertical":
            self.valid_layouts = [
                L for L in self.valid_layouts
                if getattr(L, "heuristic", "") == "single-vertical"
            ]
        elif f == "Single-horizontal":
            self.valid_layouts = [
                L for L in self.valid_layouts
                if getattr(L, "heuristic", "") == "single-horizontal"
            ]
        # else "All": no extra filtering

        if not self.valid_layouts:
            self.idx = 0
            self.canvas.delete("all")
            self.status_var.set(
                "No feasible layout found (check plot size, rooms, or 70% area cap)."
            )
            return

        self.idx = 0
        self.update_layout()

    def prev(self):
        if not self.valid_layouts:
            return
        self.idx = (self.idx - 1) % len(self.valid_layouts)
        self.update_layout()

    def next(self):
        if not self.valid_layouts:
            return
        self.idx = (self.idx + 1) % len(self.valid_layouts)
        self.update_layout()

    def draw_layout(self, layout: Layout):
        self.canvas.delete("all")

        # Scaling factors
        W, H = layout.plot_w, layout.plot_h
        scaling_factor_w = (CANVAS_W - 2 * MARGIN) / W if W > 0 else 1.0
        scaling_factor_y = (CANVAS_H - 2 * MARGIN) / H if H > 0 else 1.0
        s = min(scaling_factor_w, scaling_factor_y)

        def get_canvas_coordinate(x: float, y: float) -> tuple[float, float]:
            return (MARGIN + x * s, MARGIN + y * s)

        # Canvas fit (plot border)
        x0, y0 = get_canvas_coordinate(0, 0)
        x1, y1 = get_canvas_coordinate(W, H)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#333", width=2)

        # Build corridor list: prefer layout.corridors, fallback to layout.corridor
        corridors = getattr(layout, "corridors", None)
        if not corridors:
            c_single = getattr(layout, "corridor", None)
            corridors = [c_single] if c_single is not None else []

        # Draw every corridor (so multi-corridor layouts show up)
        for i, c in enumerate(corridors):
            cx0, cy0 = get_canvas_coordinate(c.x, c.y)
            cx1, cy1 = get_canvas_coordinate(c.x + c.width, c.y + c.height)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, fill="#dddddd", outline="#aaaaaa", width=1)
            # optional small index label
            self.canvas.create_text((cx0 + cx1) / 2, (cy0 + cy1) / 2, text=f"C{i+1}", font=("Arial", 8), fill="#666")

        # Placed rooms (draw on top)
        palette = [
            "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
            "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
            "#ccebc5", "#ffed6f",
        ]
        for idx, placed_room in enumerate(layout.placed):
            rx0, ry0 = get_canvas_coordinate(placed_room.x, placed_room.y)
            rx1, ry1 = get_canvas_coordinate(placed_room.x + placed_room.width, placed_room.y + placed_room.height)
            color = palette[idx % len(palette)]
            self.canvas.create_rectangle(rx0, ry0, rx1, ry1, fill=color, outline="#333333", width=1)
            cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
            label = f"{placed_room.name}\n{placed_room.width:.1f}×{placed_room.height:.1f}"
            self.canvas.create_text(cx, cy, text=label, font=("Arial", 10), fill="#000")

        # Legend / header: show corridor count and width
        corridor_count = len(corridors)
                # Corridor list (already handled in your updated renderer)
        corridors = getattr(layout, "corridors", None)
        corridor_count = len(corridors) if corridors else (1 if getattr(layout, "corridor", None) else 0)

        heuristic_name = getattr(layout, "heuristic", "unknown")
        legend = (
            f"Plot: {W}×{H} | Heuristic: {heuristic_name} | "
            f"Corridors: {corridor_count} × {CORRIDOR_WIDTH_UNITS} units | "
            f"Rooms area ≤ {int(AREA_FRACTION_LIMIT*100)}% of plot area"
        )

        self.canvas.create_text((x0 + x1) / 2, y0 - 10, text=legend, font=("Arial", 10), fill="#444")


    def update_layout(self):
        L: Layout = self.valid_layouts[self.idx]
        self.draw_layout(L)

        # corridor count (handle multi-corridor)
        corridors = getattr(L, "corridors", None)
        if corridors:
            corridor_count = len(corridors)
        else:
            corridor_count = 1 if getattr(L, "corridor", None) is not None else 0

        cap = AREA_FRACTION_LIMIT * L.plot_w * L.plot_h
        text = (
            f"Layout {self.idx+1}/{len(self.valid_layouts)} | {L.label} | "
            f"Placed: {L.placed_count} rooms | Rooms area: {L.rooms_area:.2f} "
            f"(cap {cap:.2f}) | Unplaced: {len(L.unplaced)} | "
            f"Corridors: {corridor_count} × {CORRIDOR_WIDTH_UNITS}"
        )

        self.status_var.set(text)


    def on_sample(self):
        sample = """\
Lobby 8 6
OfficeA 5 4
OfficeB 5 4
Meeting 6 5
Storage 4 3
Pantry 4 4
Server 3 5
ToiletM 3 3
ToiletF 3 3
OfficeC 6 4
OfficeD 6 4
OfficeE 7 3
OfficeF 7 3
"""
        self.rooms_text.delete("1.0", tk.END)
        self.rooms_text.insert("1.0", sample)
        self.plot_w_var.set("42")
        self.plot_h_var.set("28")
