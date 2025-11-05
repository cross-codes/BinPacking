import random
from random import Random
from constants import *
from ds.Layout import Layout
from ds.RoomSpec import RoomSpec
from abc import ABC, abstractmethod
from constants import *


class AbstractAlgorithm(ABC):
    def __init__(self, plot_w: float, plot_h: float, rooms: list[RoomSpec]):
        self.W: float = plot_w
        self.H: float = plot_h
        self.rooms: list[RoomSpec] = rooms
        self.rng: Random = random.Random(SEED)

    @abstractmethod
    def generate(self) -> list[Layout]:
        raise NotImplementedError
