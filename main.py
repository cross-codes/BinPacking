import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
import threading
import json
from constants import *
from ds.RoomSpec import RoomSpec
from algorithm.AbstractAlgorithm import AbstractAlgorithm
from algorithm.Backtracker import Backtracker
from algorithm.SingleCorridor import SingleCorridorAlgorithm
from algorithm.ClosedCorridorRecursive import ClosedCorridorRecursive
from algorithm.SimulatedAnnealingWrapper import SimulatedAnnealingWrapper

ALGORITHMS: dict[str, type[AbstractAlgorithm]] = {
    "Simulated Annealing (Tuned)": SimulatedAnnealingWrapper,
    "Closed Corridor Recursive": ClosedCorridorRecursive,
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
        algo_dropdown.bind("<<ComboboxSelected>>", self.on_algo_change)
        ttk.Label(top, text="Plot W:").pack(side=tk.LEFT)
        self.w_var = tk.DoubleVar(value=32.0)
        ttk.Entry(top, width=7, textvariable=self.w_var).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(top, text="H:").pack(side=tk.LEFT)
        self.h_var = tk.DoubleVar(value=36.0)
        ttk.Entry(top, width=7, textvariable=self.h_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.sa_tuned_frame = ttk.LabelFrame(top, text="SA Tuned Settings", padding=6)
        self.num_layouts_var = tk.IntVar(value=3)
        ttk.Label(self.sa_tuned_frame, text="Num Layouts:").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Entry(self.sa_tuned_frame, textvariable=self.num_layouts_var, width=8).grid(
            row=0, column=1
        )
        self.cw_var = tk.IntVar(value=5)
        ttk.Label(self.sa_tuned_frame, text="Corridor Width:").grid(
            row=1, column=0, sticky=tk.W
        )
        ttk.Entry(self.sa_tuned_frame, textvariable=self.cw_var, width=8).grid(
            row=1, column=1
        )
        self.iter_var = tk.IntVar(value=4000)
        ttk.Label(self.sa_tuned_frame, text="Iters/Restart:").grid(
            row=2, column=0, sticky=tk.W
        )
        ttk.Entry(self.sa_tuned_frame, textvariable=self.iter_var, width=8).grid(
            row=2, column=1
        )
        self.restarts_var = tk.IntVar(value=3)
        ttk.Label(self.sa_tuned_frame, text="Restarts:").grid(
            row=3, column=0, sticky=tk.W
        )
        ttk.Entry(self.sa_tuned_frame, textvariable=self.restarts_var, width=8).grid(
            row=3, column=1
        )
        self.nudge_var = tk.IntVar(value=8)
        ttk.Label(self.sa_tuned_frame, text="Nudge Budget:").grid(
            row=4, column=0, sticky=tk.W
        )
        ttk.Entry(self.sa_tuned_frame, textvariable=self.nudge_var, width=8).grid(
            row=4, column=1
        )
        self.strict_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.sa_tuned_frame, text="Strict (no nudges)", variable=self.strict_var
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W)
        self.adj_s = tk.DoubleVar(value=1200.0)
        ttk.Label(self.sa_tuned_frame, text="Adj. Reward:").grid(
            row=0, column=2, padx=(10, 0)
        )
        ttk.Scale(
            self.sa_tuned_frame,
            variable=self.adj_s,
            from_=200,
            to=2000,
            orient="horizontal",
        ).grid(row=0, column=3, sticky=tk.W)
        self.rem_s = tk.DoubleVar(value=50.0)
        ttk.Label(self.sa_tuned_frame, text="Remote Penalty:").grid(
            row=1, column=2, padx=(10, 0)
        )
        ttk.Scale(
            self.sa_tuned_frame,
            variable=self.rem_s,
            from_=0,
            to=200,
            orient="horizontal",
        ).grid(row=1, column=3, sticky=tk.W)
        self.cp_s = tk.DoubleVar(value=0.25)
        ttk.Label(self.sa_tuned_frame, text="Center Pull:").grid(
            row=2, column=2, padx=(10, 0)
        )
        ttk.Scale(
            self.sa_tuned_frame,
            variable=self.cp_s,
            from_=0.0,
            to=0.6,
            orient="horizontal",
        ).grid(row=2, column=3, sticky=tk.W)
        self.generate_btn = ttk.Button(top, text="Generate", command=self.generate)
        self.generate_btn.pack(side=tk.RIGHT, padx=6)

        self.export_btn = ttk.Button(
            top, text="Export JSON", command=self.export_json, state="disabled"
        )
        self.export_btn.pack(side=tk.RIGHT, padx=6)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        ttk.Label(left, text="Rooms (Name W H [interchangeable 1/0]):").pack(
            anchor=tk.W
        )
        self.text = ScrolledText(left, width=30, height=28)
        self.text.pack(fill=tk.Y)
        self.text.insert(
            "1.0",
            "MeetingRm 9 9 1\nCabin1 6 7 1\nCabin2 6 7 1\nCabin3 6 7 1\nMdCabin1 10 12 1\nMdCabin2 10 12 1\nWorkstations1 8 6 1\nWorkstations2 8 6 1\nWorkstations3 8 6 1\nConfRoom 10 15 1\nOpenStorage 5 7 1\nPantry 6 7 1\nWC 5 6 1\n",
        )
        self.canvas = tk.Canvas(main, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.prev_btn = ttk.Button(
            bottom, text="<< Prev", command=self.prev, state="disabled"
        )
        self.prev_btn.pack(side=tk.LEFT, padx=6)
        self.next_btn = ttk.Button(
            bottom, text="Next >>", command=self.next, state="disabled"
        )
        self.next_btn.pack(side=tk.LEFT)
        self.info = ttk.Label(bottom, text="No layouts")
        self.info.pack(side=tk.LEFT, padx=10)
        self.status = ttk.Label(bottom, text="Ready")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        self.layouts, self.idx, self.total_rooms = [], 0, 0
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.has_generated_once = False
        self.on_algo_change()
        self.redraw()

    def on_algo_change(self, event=None):
        if self.algo_var.get() == "Simulated Annealing (Tuned)":
            self.sa_tuned_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        else:
            self.sa_tuned_frame.pack_forget()

    def generate(self):
        self.has_generated_once = True
        self.status.config(text="Generating...", foreground="black")
        self.generate_btn.config(state="disabled")
        self.prev_btn.config(state="disabled")
        self.next_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.root.update_idletasks()
        rooms = self.parse_rooms()
        self.total_rooms = len(rooms)
        if not rooms:
            self.status.config(text="No rooms.", foreground="red")
            self.generate_btn.config(state="normal")
            return
        algo_name, plot_w, plot_h = (
            self.algo_var.get(),
            self.w_var.get(),
            self.h_var.get(),
        )
        sa_params = {}
        if algo_name == "Simulated Annealing (Tuned)":
            sa_params = {
                "num_layouts": self.num_layouts_var.get(),
                "corridor_width": self.cw_var.get(),
                "total_iters": self.iter_var.get(),
                "restarts": self.restarts_var.get(),
                "nudges_budget": self.nudge_var.get(),
                "strict_mode": self.strict_var.get(),
                "adj_reward": self.adj_s.get(),
                "remote_penalty": self.rem_s.get(),
                "center_pull_prob": self.cp_s.get(),
            }
        threading.Thread(
            target=self._worker_thread,
            args=(algo_name, plot_w, plot_h, rooms, sa_params),
            daemon=True,
        ).start()

    def _worker_thread(self, algo_name, plot_w, plot_h, rooms, sa_params):
        try:
            AlgorithmClass = ALGORITHMS[algo_name]
            gen = (
                AlgorithmClass(plot_w, plot_h, rooms, **sa_params)
                if sa_params
                else AlgorithmClass(plot_w, plot_h, rooms)
            )
            layouts = gen.generate()
            self.root.after(0, self._generation_complete, layouts)
        except Exception as e:
            import traceback

            print(f"Error in worker: {e}")
            traceback.print_exc()
            self.root.after(0, self._generation_complete, [])

    def _generation_complete(self, layouts):
        self.layouts = [l for l in layouts if l is not None]
        self.idx = 0
        self.status.config(text=f"Found {len(self.layouts)} layouts.")
        self.generate_btn.config(state="normal")
        if self.layouts:
            self.prev_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.export_btn.config(state="normal")
        self.redraw()

    def export_json(self):
        if not self.layouts:
            return

        layout_data = self.layouts[self.idx]
        export_data = {}

        if isinstance(layout_data, dict):
            backbone = layout_data.get("backbone")
            backbone_list = list(backbone) if backbone is not None else []

            export_data = {
                "algorithm": "Simulated Annealing",
                "plot_width": int(self.w_var.get()),
                "plot_height": int(self.h_var.get()),
                "energy": layout_data.get("energy", 0),
                "nudges": layout_data.get("nudges", 0),
                "rooms": [
                    {
                        "name": r.name,
                        "x": r.x,
                        "y": r.y,
                        "w": r.w,
                        "h": r.h,
                        "rotated": getattr(r, "rotated", False)
                        or getattr(r, "rot", False),
                    }
                    for r in layout_data["rooms"]
                ],
                "backbone": backbone_list,  # Use the list version
            }
        else:
            backbone = getattr(layout_data, "backbone", None)
            backbone_list = list(backbone) if backbone is not None else []

            export_data = {
                "algorithm": self.algo_var.get(),
                "plot_width": layout_data.plot_w,
                "plot_height": layout_data.plot_h,
                "rooms": [
                    {
                        "name": r.name,
                        "x": r.x,
                        "y": r.y,
                        "w": r.w,
                        "h": r.h,
                        "rotated": getattr(r, "rotated", False)
                        or getattr(r, "rot", False),
                    }
                    for r in layout_data.placed
                ],
                "corridor": (
                    {
                        "x": layout_data.corridor.x,
                        "y": layout_data.corridor.y,
                        "w": layout_data.corridor.w,
                        "h": layout_data.corridor.h,
                    }
                    if layout_data.corridor
                    else None
                ),
                "backbone": backbone_list, # Use the list version
            }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Layout",
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(export_data, f, indent=4)
                self.status.config(text=f"Saved to {file_path}", foreground="green")
            except Exception as e:
                self.status.config(text=f"Error saving: {e}", foreground="red")

    def prev(self):
        if self.layouts:
            self.idx = (self.idx - 1) % len(self.layouts)
            self.redraw()

    def next(self):
        if self.layouts:
            self.idx = (self.idx + 1) % len(self.layouts)
            self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.layouts:
            self.info.config(
                text=(
                    "No layouts found."
                    if self.has_generated_once
                    else "Ready to generate."
                )
            )
            self.status.config(text="Adjust settings and click 'Generate'.")
            return

        layout_data = self.layouts[self.idx]
        is_new_algo = isinstance(layout_data, dict)

        if is_new_algo:
            plot_w, plot_h = self.w_var.get(), self.h_var.get()
            placed_rooms, backbone, corridor = (
                layout_data["rooms"],
                layout_data.get("backbone"),
                None,
            )
            unplaced_names, placed_count = [], len(placed_rooms)
            info_text = f"Layout {self.idx+1}/{len(self.layouts)} (Energy: {layout_data.get('energy',0):.1f})"
            status_text = "Generated via SA (Tuned)"
        else:
            plot_w, plot_h = layout_data.plot_w, layout_data.plot_h
            placed_rooms, backbone, corridor = (
                layout_data.placed,
                layout_data.backbone,
                layout_data.corridor,
            )
            unplaced_names, placed_count = (
                layout_data.unplaced_names,
                layout_data.placed_count,
            )
            info_text = f"Layout {self.idx+1}/{len(self.layouts)} ; Placed {placed_count}/{self.total_rooms}"
            status_text = (
                "All rooms placed."
                if not unplaced_names
                else f"Unplaced: {', '.join(layout_data.unplaced_names)}"
            )

        pad = 60
        W, H = plot_w, plot_h
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        scale = min((cw - 2 * pad) / W, (ch - 2 * pad) / H) if W * H > 0 else 1
        ox, oy = (cw - W * scale) / 2, (ch - H * scale) / 2

        # Draw 1-unit Grid
        grid_color = "#e8e8e8"  # Very light gray for background grid
        for i in range(int(W) + 1):
            x = ox + i * scale
            self.canvas.create_line(x, oy, x, oy + H * scale, fill=grid_color, width=1)
        for i in range(int(H) + 1):
            y = oy + i * scale
            self.canvas.create_line(ox, y, ox + W * scale, y, fill=grid_color, width=1)

        # Draw Main Boundary (removed fill so grid is visible)
        self.canvas.create_rectangle(
            ox, oy, ox + W * scale, oy + H * scale, fill="", outline=""
        )

        corridor_fill = "#d0d0d0"
        if backbone:
            for cx, cy in backbone:
                y_start_flipped = plot_h - cy - 1
                self.canvas.create_rectangle(
                    ox + cx * scale,
                    oy + y_start_flipped * scale,
                    ox + (cx + 1) * scale,
                    oy + (y_start_flipped + 1) * scale,
                    fill=corridor_fill,
                    outline="",
                )
        elif corridor:
            self.canvas.create_rectangle(
                ox + corridor.x * scale,
                oy + corridor.y * scale,
                ox + (corridor.x + corridor.w) * scale,
                oy + (corridor.y + corridor.h) * scale,
                fill=corridor_fill,
                outline="",
            )

        for r in placed_rooms:
            y_start = (plot_h - r.y - r.h) if is_new_algo else r.y
            x1, y1, x2, y2 = (
                ox + r.x * scale,
                oy + y_start * scale,
                ox + (r.x + r.w) * scale,
                oy + (y_start + r.h) * scale,
            )
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="#4ea3ff", outline="#003366", width=1.5
            )
            name_tag = (
                f"{r.name}*"
                if (hasattr(r, "rot") and r.rot)
                or (hasattr(r, "rotated") and r.rotated)
                else r.name
            )
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
        self._draw_total_dimension_lines(plot_w, plot_h, scale, ox, oy)

        boundary_segs = self._calculate_free_boundary_segments(
            plot_w, plot_h, placed_rooms, is_new_algo
        )
        self._draw_segment_dimension_lines(boundary_segs, plot_w, plot_h, scale, ox, oy)

        self.info.config(text=info_text)
        self.status.config(
            text=status_text, foreground="green" if not unplaced_names else "red"
        )

    def _calculate_free_boundary_segments(self, plot_w, plot_h, rooms, is_new_algo):
        EPS = 1e-6

        def subtract_interval(intervals, sub):
            res = []
            for i in intervals:
                if sub[1] <= i[0] or sub[0] >= i[1]:
                    res.append(i)
                    continue
                if sub[0] > i[0]:
                    res.append((i[0], sub[0]))
                if sub[1] < i[1]:
                    res.append((sub[1], i[1]))
            return res

        segs = {
            "top": [(0, plot_w)],
            "bottom": [(0, plot_w)],
            "left": [(0, plot_h)],
            "right": [(0, plot_h)],
        }
        for r in rooms:
            room_y = (plot_h - r.y - r.h) if is_new_algo else r.y

            if abs(room_y) < EPS:  # Room on top edge
                segs["top"] = subtract_interval(segs["top"], (r.x, r.x + r.w))
            if abs(room_y + r.h - plot_h) < EPS:  # Room on bottom edge
                segs["bottom"] = subtract_interval(segs["bottom"], (r.x, r.x + r.w))
            if abs(r.x) < EPS:  # Room on left edge
                segs["left"] = subtract_interval(segs["left"], (room_y, room_y + r.h))
            if abs(r.x + r.w - plot_w) < EPS:  # Room on right edge
                segs["right"] = subtract_interval(segs["right"], (room_y, room_y + r.h))

        return segs

    def _draw_total_dimension_lines(self, plot_w, plot_h, scale, ox, oy):
        W, H = plot_w, plot_h
        font_size = 8
        offset = 15
        tick_size = 3
        text_padding = 3
        total_dim_offset = offset + 27
        total_line_color = "#00008B"
        x1, x2, y_pos = ox, ox + W * scale, oy - total_dim_offset
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
        y1, y2, x_pos = oy, oy + H * scale, ox - total_dim_offset
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

    def _draw_segment_dimension_lines(self, segs, plot_w, plot_h, scale, ox, oy):
        W, H = plot_w, plot_h
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
        for side, segments in segs.items():
            for start, end in segments:
                if end - start > min_len_to_show:
                    if side == "top" or side == "bottom":
                        x1, x2 = ox + start * scale, ox + end * scale
                        y_pos = (
                            oy - offset if side == "top" else oy + H * scale + offset
                        )
                        self.canvas.create_line(
                            x1,
                            y_pos - tick_size,
                            x1,
                            y_pos + tick_size,
                            fill=line_color,
                        )
                        self.canvas.create_line(
                            x2,
                            y_pos - tick_size,
                            x2,
                            y_pos + tick_size,
                            fill=line_color,
                        )
                        self.canvas.create_line(x1, y_pos, x2, y_pos, fill=line_color)
                        self.canvas.create_text(
                            (x1 + x2) / 2,
                            (
                                y_pos - text_padding
                                if side == "top"
                                else y_pos + text_padding
                            ),
                            text=f"{end-start:.1f}",
                            font=("Arial", font_size),
                            fill=line_color,
                            anchor=tk.S if side == "top" else tk.N,
                        )
                    else:
                        y1, y2 = oy + start * scale, oy + end * scale
                        x_pos = (
                            ox - offset if side == "left" else ox + W * scale + offset
                        )
                        self.canvas.create_line(
                            x_pos - tick_size,
                            y1,
                            x_pos + tick_size,
                            y1,
                            fill=line_color,
                        )
                        self.canvas.create_line(
                            x_pos - tick_size,
                            y2,
                            x_pos + tick_size,
                            y2,
                            fill=line_color,
                        )
                        self.canvas.create_line(x_pos, y1, x_pos, y2, fill=line_color)
                        self.canvas.create_text(
                            (
                                x_pos - text_padding
                                if side == "left"
                                else x_pos + text_padding
                            ),
                            (y1 + y2) / 2,
                            text=f"{end-start:.1f}",
                            font=("Arial", font_size),
                            fill=line_color,
                            anchor=tk.E if side == "left" else tk.W,
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


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
