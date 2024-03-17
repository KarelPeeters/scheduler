import math
from typing import TypeVar, Generic, Callable, Set, List

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
        self.min_time = math.inf
        self.min_time_actions = None

        self.min_energy = math.inf
        self.min_energy_actions = None

        with open("log.txt", "w"):
            # clear log
            pass

    def add_solution(self, time: float, energy: float, actions: List):
        either = False

        with open("log.txt", "a") as f:
            if time < self.min_time or time == self.min_time and energy < self.min_energy:
                either = True
                self.min_time = time
                self.min_time_actions = actions
                print(f"New best time solution: ({time}, {energy})", file=f)

            if energy < self.min_energy or energy == self.min_energy and time < self.min_time:
                either = True
                self.min_energy = energy
                self.min_energy_actions = actions
                print(f"New best energy solution: ({time}, {energy})", file=f)

            if either:
                for action in actions:
                    print(f"  {action}", file=f)

    def is_dominated(self, time: float, energy: float):
        return time >= self.min_time and energy >= self.min_energy
