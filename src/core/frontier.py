import math
from typing import TypeVar, Generic, Callable, Set, List, Dict

import matplotlib.pyplot as plt

T = TypeVar("T")


class ParetoFrontier(Generic[T]):
    """ Stores a pareto frontier of _maximized_ items. """

    def __init__(self, dominates: Callable[[T, T], bool]):
        self.frontier: Set[T] = set()
        self.dominates = dominates

    def __len__(self):
        return len(self.frontier)

    def add(self, new: T) -> bool:
        """
        Try adding the given state to the frontier.
        Returns whether this state improves on the current frontier and was indeed added.
        """
        to_remove = set()

        for old in self.frontier:
            if self.dominates(old, new):
                return False
            if self.dominates(new, old):
                to_remove.add(old)
                # TODO can we break here? only if "dominates" is transitive, do we guarantee that?
                # TODO collect all values ever seen and then do some transitive and non-reflexive checks

        # TODO remove?
        # assert len(to_remove) <= 1, "remove and document that this assumption is not actually right"

        for old in to_remove:
            self.frontier.remove(old)

        self.frontier.add(new)
        return True


class SimpleFrontier:
    def __init__(self):
        self.all_pairs = set()
        self.best_pairs = set()

        with open("log.txt", "w"):
            # clear log
            pass

    def add_solution(self, time: float, energy: float, actions: List):
        to_remove = set()
        to_add = True

        for t, e in self.best_pairs:
            dominates = (time < t or energy < e) and (time <= t and energy <= e)
            is_dominated = (t < time or e < energy) and (t <= time and e <= energy)

            if dominates:
                to_remove.add((t, e))
            if is_dominated:
                to_add = False

        for p in to_remove:
            self.best_pairs.remove(p)

        self.all_pairs.add((time, energy))
        if to_add:
            self.best_pairs.add((time, energy))

            with open("log.txt", "a") as f:
                print(f"New pareto solution: ({time}, {energy})", file=f)
                for action in actions:
                    print(f"  {action}", file=f)

        # update scatter plot
        # TODO maybe update this more often?
        fig, ax = plt.subplots()

        all_times = []
        all_energies = []
        all_colors = []
        for t, e in self.all_pairs:
            all_times.append(t)
            all_energies.append(e)
            all_colors.append((t, e) in self.best_pairs)

        ax.scatter(x=all_times, y=all_energies, c=all_colors)
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        fig.savefig("../ignored/pairs.png")

    def is_dominated(self, time: float, energy: float):
        for t, e in self.best_pairs:
            if (t < time or e < energy) and (t <= time and e <= energy):
                return True
        return False
