from algorithm.AbstractAlgorithm import AbstractAlgorithm
from ds.RoomSpec import RoomSpec

from algorithm.SA5 import Room, run_sa_with_restarts, energy_of_layout


class SimulatedAnnealingWrapper(AbstractAlgorithm):
    def __init__(
        self,
        plot_w: float,
        plot_h: float,
        rooms: list[RoomSpec],
        num_layouts: int,
        corridor_width: int,
        total_iters: int,
        restarts: int,
        nudges_budget: int,
        strict_mode: bool,
        adj_reward: float,
        remote_penalty: float,
        center_pull_prob: float,
    ):
        super().__init__(plot_w, plot_h, rooms)
        self.num_layouts = num_layouts
        self.corridor_width = corridor_width
        self.total_iters = total_iters
        self.restarts = restarts
        self.nudges_budget = nudges_budget
        self.strict_mode = strict_mode
        self.adj_reward = adj_reward
        self.remote_penalty = remote_penalty
        self.center_pull_prob = center_pull_prob

    def generate(self) -> list[dict]:
        rooms_internal = [
            Room(w=spec.w, h=spec.h, idx=i, name=spec.name)
            for i, spec in enumerate(self.rooms)
        ]

        generated_layouts = []
        for i in range(self.num_layouts):
            print(f"--- Starting SA Generation Run {i+1}/{self.num_layouts} ---")

            result = run_sa_with_restarts(
                plot_w=int(self.W),
                plot_h=int(self.H),
                rooms_input=rooms_internal,
                corridor_width=self.corridor_width,
                max_total_iters=self.total_iters,
                restarts=self.restarts,
                nudges_budget=self.nudges_budget,
                strict_mode=self.strict_mode,
                center_pull_prob=self.center_pull_prob,
            )

            if result and result.get("success"):
                result["energy"] = energy_of_layout(
                    result["rooms"],
                    int(self.W),
                    int(self.H),
                    result["backbone"],
                    result.get("nudges", 0),
                    0.7 * (self.W * self.H),
                    adjacency_reward=self.adj_reward,
                    remote_penalty=self.remote_penalty,
                )
                generated_layouts.append(result)

        return generated_layouts
