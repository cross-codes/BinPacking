from algorithm.AbstractAlgorithm import AbstractAlgorithm
from ds.RoomSpec import RoomSpec

from algorithm.original_sa_script import Room, run_sa_with_restarts, energy_of_layout

class SimulatedAnnealingWrapper(AbstractAlgorithm):
    """
    This class is a wrapper around the original, unmodified SA script.
    Its only job is to translate data to and from the script's native format.
    It returns a list of dictionaries, NOT Layout objects.
    """
    def __init__(
        self,
        plot_w: float,
        plot_h: float,
        rooms: list[RoomSpec],
        # Parameters from the UI
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
        # 1. Convert your app's RoomSpec into the script's internal Room class
        rooms_internal = [Room(w=spec.w, h=spec.h, idx=i, name=spec.name) for i, spec in enumerate(self.rooms)]

        generated_layouts = []
        for i in range(self.num_layouts):
            print(f"--- Starting SA Generation Run {i+1}/{self.num_layouts} ---")
            
            # 2. Call the ORIGINAL, UNMODIFIED run_sa_with_restarts function
            result = run_sa_with_restarts(
                plot_w=int(self.W),
                plot_h=int(self.H),
                rooms_input=rooms_internal,
                corridor_width=self.corridor_width,
                max_total_iters=self.total_iters,
                restarts=self.restarts,
                nudges_budget=self.nudges_budget,
                strict_mode=self.strict_mode,
                center_pull_prob=self.center_pull_prob
            )

            if result and result.get('success'):
                # 3. Recalculate energy using UI slider values for consistent display
                # This was part of the original script's GUI logic and is necessary.
                result['energy'] = energy_of_layout(
                    result['rooms'], int(self.W), int(self.H), result['backbone'],
                    result.get('nudges', 0), 0.7 * (self.W * self.H),
                    adjacency_reward=self.adj_reward, remote_penalty=self.remote_penalty
                )
                generated_layouts.append(result)

        # 4. Return the raw dictionary output. The main app will handle drawing it.
        return generated_layouts
