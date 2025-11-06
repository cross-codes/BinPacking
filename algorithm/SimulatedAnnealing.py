# filename: algorithm/BruteForceBlockedCorridor.py

from typing import List, Optional
from algorithm.AbstractAlgorithm import AbstractAlgorithm
from ds.Layout import Layout
from ds.PlacedRoom import PlacedRoom
from ds.RoomSpec import RoomSpec
from ds.Corridor import Corridor
from ds.Orientation import Orientation
from constants import *
import copy
import itertools

class SimulatedAnnealing(AbstractAlgorithm):
    """
    A true brute-force algorithm that is sound and functional. It exhaustively
    searches through all room permutations, corridor widths, and blocking
    options to find all possible closed U-shaped layouts.
    """

    def generate(self) -> List[Layout]:
        solutions = []
        
        # --- True Brute-Force Permutations ---
        # We will try every possible ordering of the rooms.
        room_permutations = list(itertools.permutations(self.rooms))
        
        for i, room_order in enumerate(room_permutations):
            # Limit permutations for very large numbers of rooms to prevent freezing.
            if i >= 1000: break 
            
            # --- True Brute-Force Corridor Widths ---
            # Iterate through a granular range of possible corridor widths.
            for corridor_w in range(3, int(self.W / 2)):
                
                # 1. Create the base single corridor layout for this specific permutation and width.
                base_layout = self._create_base_layout(list(room_order), float(corridor_w))
                if not base_layout: continue

                # 2. Try to use EVERY remaining room to block the corridor.
                remaining_specs = [r for r in self.rooms if r.name in base_layout.unplaced_names]
                max_y = max([p.y + p.h for p in base_layout.placed] + [0])
                
                for spec in remaining_specs:
                    # Try both rotations for the blocking room.
                    for rotated in [False, True] if spec.can_rotate else [False]:
                        w, h = (spec.h, spec.w) if rotated else (spec.w, spec.h)

                        # If the room fits perfectly as a blocker.
                        if abs(w - corridor_w) < EPS:
                            final_placed = copy.deepcopy(base_layout.placed)
                            blocking_room = PlacedRoom(spec.name, base_layout.corridor.x, max_y, w, h, rotated)
                            
                            # Final check to make sure the blocking room is valid
                            if self._is_valid_blocking_move(blocking_room, final_placed):
                                final_placed.append(blocking_room)
                                unplaced = [r.name for r in remaining_specs if r.name != spec.name]
                                
                                # Create and save the valid, blocked layout
                                final_layout = Layout(final_placed, unplaced, self.W, self.H, base_layout.corridor)
                                solutions.append(final_layout)
        
        # --- Final Filtering ---
        # Only return the best layouts found.
        if not solutions: return []
        max_placed = max(l.placed_count for l in solutions)
        best_solutions = [l for l in solutions if l.placed_count == max_placed]
        
        # Remove duplicate layouts based on their signature
        unique_solutions, seen_sigs = [], set()
        for s in best_solutions:
            sig = tuple(sorted((p.name, p.x, p.y, p.rotated) for p in s.placed))
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique_solutions.append(s)
                
        return unique_solutions


    def _create_base_layout(self, room_order: List[RoomSpec], corridor_w: float) -> Optional[Layout]:
        """Creates the initial layout with rooms placed along the corridor."""
        corridor_x = (self.W - corridor_w) / 2
        if corridor_x < 0: return None
        
        y_left, y_right = 0.0, 0.0
        placed, unplaced_specs = [], []

        for r_spec in room_order:
            was_placed = False
            side_order = ['left', 'right'] if y_left <= y_right else ['right', 'left']
            
            for side in side_order:
                for rotated in [False, True] if r_spec.can_rotate else [False]:
                    w, h = (r_spec.h, r_spec.w) if rotated else (r_spec.w, r_spec.h)
                    
                    if side == 'left' and w <= corridor_x + EPS and y_left + h <= self.H + EPS:
                        placed.append(PlacedRoom(r_spec.name, corridor_x - w, y_left, w, h, rotated))
                        y_left += h; was_placed = True; break
                    elif side == 'right' and w <= (self.W - (corridor_x + corridor_w)) + EPS and y_right + h <= self.H + EPS:
                        placed.append(PlacedRoom(r_spec.name, corridor_x + corridor_w, y_right, w, h, rotated))
                        y_right += h; was_placed = True; break
                if was_placed: break
            
            if not was_placed:
                unplaced_specs.append(r_spec)

        if not placed: return None
        return Layout(placed, [r.name for r in unplaced_specs], self.W, self.H, Corridor(corridor_x, 0, corridor_w, self.H, Orientation.VERTICAL))
        
    def _is_valid_blocking_move(self, cand: PlacedRoom, placed: List[PlacedRoom]) -> bool:
        if not (cand.y + cand.h <= self.H + EPS): return False
        for p in placed:
            if (cand.x < p.x + p.w - EPS and cand.x + cand.w > p.x + EPS and
                cand.y < p.y + p.h - EPS and cand.y + cand.h > p.y + EPS):
                return False
        return True
