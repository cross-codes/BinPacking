import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from constants import *
from ds.RoomSpec import RoomSpec
from algorithm.AbstractAlgorithm import AbstractAlgorithm
from algorithm.Backtracker import Backtracker
from algorithm.SingleCorridor import SingleCorridorAlgorithm
from algorithm.SimulatedAnnealing import SimulatedAnnealing

ALGORITHMS: dict[str, type[AbstractAlgorithm]] = {
    "Simulated Annealing": SimulatedAnnealing,
    "Backtracking": Backtracker,
    "Single Corridor": SingleCorridorAlgorithm,
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Layout Generator")

        top = ttk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        main = ttk.Frame(root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        bottom = ttk.Frame(root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=4)

        ttk.Label(top, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value=list(ALGORITHMS.keys())[0])

        algo_dropdown = ttk.Combobox(
            top,
            textvariable=self.algo_var,
            values=list(ALGORITHMS.keys()),
            state="readonly",
        )
        algo_dropdown.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(top, text="Plot W:").pack(side=tk.LEFT)
        self.w_var = tk.DoubleVar(value=20.0)
        ttk.Entry(top, width=7, textvariable=self.w_var).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(top, text="H:").pack(side=tk.LEFT)
        self.h_var = tk.DoubleVar(value=25.0)
        ttk.Entry(top, width=7, textvariable=self.h_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        ttk.Button(top, text="Generate", command=self.generate).pack(
            side=tk.RIGHT, padx=6
        )

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        ttk.Label(left, text="Rooms (Name W H [interchangeable 1/0]):").pack(anchor=tk.W)
        self.text = ScrolledText(left, width=30, height=28)
        self.text.pack(fill=tk.Y)
        self.text.insert(
            "1.0",
            "LivingRm 6 5 1\nKitchen 7 4 1\nBed1 4 4 0\nBed2 5 5 0\nBath1 3 8 1\nDining 6 6 0\nOffice 3 9 1",
        )
        self.canvas = tk.Canvas(main, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ttk.Button(bottom, text="<< Prev", command=self.prev).pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom, text="Next >>", command=self.next).pack(side=tk.LEFT)
        self.info = ttk.Label(bottom, text="No layouts")
        self.info.pack(side=tk.LEFT, padx=10)
        self.status = ttk.Label(bottom, text="Ready")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6)

        self.layouts, self.idx, self.total_rooms = [], 0, 0
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.generate()

    def generate(self):
        self.status.config(text="Generating...", foreground="black")
        self.root.update_idletasks()
        rooms = self.parse_rooms()
        self.total_rooms = len(rooms)
        if not rooms:
            self.status.config(text="No rooms.", foreground="red")
            return

        AlgorithmClass = ALGORITHMS[self.algo_var.get()]
        gen = AlgorithmClass(self.w_var.get(), self.h_var.get(), rooms)
        self.layouts = gen.generate()
        self.idx = 0

        self.status.config(text=f"Found {len(self.layouts)} valid layouts.")
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts:
            self.info.config(text="No valid layouts.")
            self.status.config(
                text="Try a larger plot or different algorithm.", foreground="red"
            )
            return

        layout = self.layouts[self.idx]
        pad = 60
        W, H = layout.plot_w, layout.plot_h
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        scale = min((cw - 2 * pad) / W, (ch - 2 * pad) / H) if W * H > 0 else 1
        ox, oy = (cw - W * scale) / 2, (ch - H * scale) / 2

        if layout.corridor:
            c = layout.corridor
            self.canvas.create_rectangle(
                ox + c.x * scale,
                oy + c.y * scale,
                ox + (c.x + c.w) * scale,
                oy + (c.y + c.h) * scale,
                fill="#d0d0d0",
                outline="",
            )
        else:
            self.canvas.create_rectangle(
                ox, oy, ox + W * scale, oy + H * scale, fill="#d0d0d0", outline=""
            )

        for r in layout.placed:
            x1, y1, x2, y2 = (
                ox + r.x * scale,
                oy + r.y * scale,
                ox + (r.x + r.w) * scale,
                oy + (r.y + r.h) * scale,
            )
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="#4ea3ff", outline="#003366", width=1.5
            )
            name_tag = f"{r.name}*" if r.rotated else r.name
            self.canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=f"{name_tag}\n{r.w:.1f}x{r.h:.1f}",
                fill="white",
                font=("Arial", 9),
            )

        self.canvas.create_rectangle(
            ox, oy, ox + W * scale, oy + H * scale, width=2, outline="#333"
        )

        if not layout.corridor:
            self._draw_segment_dimension_lines(layout, scale, ox, oy)

        self._draw_total_dimension_lines(layout, scale, ox, oy)

        if layout.entrance_pos:
            ex, ey = (
                ox + layout.entrance_pos[0] * scale,
                oy + layout.entrance_pos[1] * scale,
            )
            is_vertical_edge = (
                abs(layout.entrance_pos[0]) < EPS
                or abs(layout.entrance_pos[0] - W) < EPS
            )
            dx, dy = (0, 10) if is_vertical_edge else (10, 0)
            self.canvas.create_line(
                ex - dx, ey - dy, ex + dx, ey + dy, fill="red", width=5
            )

        self.info.config(
            text=f"Layout {self.idx+1}/{len(self.layouts)} ; Placed {layout.placed_count}/{self.total_rooms}"
        )
        status_text = (
            "All rooms placed."
            if not layout.unplaced_names
            else f"Unplaced: {', '.join(layout.unplaced_names)}"
        )
        self.status.config(
            text=status_text, foreground="green" if not layout.unplaced_names else "red"
        )

    def _draw_segment_dimension_lines(self, layout, scale, ox, oy):
        W, H = layout.plot_w, layout.plot_h
        font_size = 8
        offset = 15
        tick_size = 3
        min_len_to_show = 0.5
        line_color = "#555555"
        text_padding = 3

        self.canvas.create_line(ox - 5, oy - 5, ox + 5, oy + 5, fill="red", width=1.5)
        self.canvas.create_line(ox - 5, oy + 5, ox + 5, oy - 5, fill="red", width=1.5)
        self.canvas.create_text(
            ox - 8,
            oy - 8,
            text="(0,0)",
            font=("Arial", font_size),
            fill="red",
            anchor=tk.SE,
        )

        segs = layout.free_boundary_segments
        if segs:
            for start, end in segs.get("top", []):
                length = end - start
                if length > min_len_to_show:
                    x1, x2 = ox + start * scale, ox + end * scale
                    y_pos = oy - offset
                    self.canvas.create_line(
                        x1, y_pos - tick_size, x1, y_pos + tick_size, fill=line_color
                    )
                    self.canvas.create_line(
                        x2, y_pos - tick_size, x2, y_pos + tick_size, fill=line_color
                    )
                    self.canvas.create_line(x1, y_pos, x2, y_pos, fill=line_color)
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        y_pos - text_padding,
                        text=f"{length:.1f}",
                        font=("Arial", font_size),
                        fill=line_color,
                        anchor=tk.S,
                    )

            for start, end in segs.get("bottom", []):
                length = end - start
                if length > min_len_to_show:
                    x1, x2 = ox + start * scale, ox + end * scale
                    y_pos = oy + H * scale + offset
                    self.canvas.create_line(
                        x1, y_pos - tick_size, x1, y_pos + tick_size, fill=line_color
                    )
                    self.canvas.create_line(
                        x2, y_pos - tick_size, x2, y_pos + tick_size, fill=line_color
                    )
                    self.canvas.create_line(x1, y_pos, x2, y_pos, fill=line_color)
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        y_pos + text_padding,
                        text=f"{length:.1f}",
                        font=("Arial", font_size),
                        fill=line_color,
                        anchor=tk.N,
                    )

            for start, end in segs.get("left", []):
                length = end - start
                if length > min_len_to_show:
                    y1, y2 = oy + start * scale, oy + end * scale
                    x_pos = ox - offset
                    self.canvas.create_line(
                        x_pos - tick_size, y1, x_pos + tick_size, y1, fill=line_color
                    )
                    self.canvas.create_line(
                        x_pos - tick_size, y2, x_pos + tick_size, y2, fill=line_color
                    )
                    self.canvas.create_line(x_pos, y1, x_pos, y2, fill=line_color)
                    self.canvas.create_text(
                        x_pos - text_padding,
                        (y1 + y2) / 2,
                        text=f"{length:.1f}",
                        font=("Arial", font_size),
                        fill=line_color,
                        anchor=tk.E,
                    )

            for start, end in segs.get("right", []):
                length = end - start
                if length > min_len_to_show:
                    y1, y2 = oy + start * scale, oy + end * scale
                    x_pos = ox + W * scale + offset
                    self.canvas.create_line(
                        x_pos - tick_size, y1, x_pos + tick_size, y1, fill=line_color
                    )
                    self.canvas.create_line(
                        x_pos - tick_size, y2, x_pos + tick_size, y2, fill=line_color
                    )
                    self.canvas.create_line(x_pos, y1, x_pos, y2, fill=line_color)
                    self.canvas.create_text(
                        x_pos + text_padding,
                        (y1 + y2) / 2,
                        text=f"{length:.1f}",
                        font=("Arial", font_size),
                        fill=line_color,
                        anchor=tk.W,
                    )

    def _draw_total_dimension_lines(self, layout, scale, ox, oy):
        W, H = layout.plot_w, layout.plot_h
        font_size = 8
        offset = 15
        tick_size = 3
        text_padding = 3
        total_dim_offset = offset + 27
        total_line_color = "#00008B"

        x1, x2 = ox, ox + W * scale
        y_pos = oy - total_dim_offset
        self.canvas.create_line(
            x1, y_pos - tick_size, x1, y_pos + tick_size, fill=total_line_color
        )
        self.canvas.create_line(
            x2, y_pos - tick_size, x2, y_pos + tick_size, fill=total_line_color
        )
        self.canvas.create_line(x1, y_pos, x2, y_pos, fill=total_line_color)
        self.canvas.create_text(
            (x1 + x2) / 2,
            y_pos - text_padding,
            text=f"W = {W:.1f}",
            font=("Arial", font_size, "bold"),
            fill=total_line_color,
            anchor=tk.S,
        )

        y1, y2 = oy, oy + H * scale
        x_pos = ox - total_dim_offset
        self.canvas.create_line(
            x_pos - tick_size, y1, x_pos + tick_size, y1, fill=total_line_color
        )
        self.canvas.create_line(
            x_pos - tick_size, y2, x_pos + tick_size, y2, fill=total_line_color
        )
        self.canvas.create_line(x_pos, y1, x_pos, y2, fill=total_line_color)
        self.canvas.create_text(
            x_pos - text_padding,
            (y1 + y2) / 2,
            text=f"H = {H:.1f}",
            font=("Arial", font_size, "bold"),
            fill=total_line_color,
            anchor=tk.E,
        )

    def parse_rooms(self) -> list[RoomSpec]:
        rooms = []
        for line in self.text.get("1.0", tk.END).strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    can_rotate = len(parts) > 3 and parts[3] in [
                        "1",
                        "yes",
                        "true",
                        "rotatable",
                    ]
                    rooms.append(
                        RoomSpec(parts[0], float(parts[1]), float(parts[2]), can_rotate)
                    )
                except ValueError:
                    continue
        return rooms

    def prev(self):
        if self.layouts:
            self.idx = (self.idx - 1 + len(self.layouts)) % len(self.layouts)
            self.redraw()

    def next(self):
        if self.layouts:
            self.idx = (self.idx + 1) % len(self.layouts)
            self.redraw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
