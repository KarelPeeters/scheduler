from typing import TypeVar, Generic, Callable, Set, Tuple

import matplotlib.pyplot as plt

T = TypeVar("T")


class ParetoFrontier(Generic[T]):
    """ Stores a pareto frontier of _maximized_ items. """

    def __init__(self, dominates: Callable[[T, T], bool]):
        self.frontier: Set[T] = set()
        self.dominates = dominates

        # TODO remove
        self.all: Set[T] = set()

    def __len__(self):
        return len(self.frontier)

    def add(self, new: T) -> bool:
        """
        Try adding the given state to the frontier.
        Returns whether this state improves on the current frontier and was indeed added.
        """
        self.all.add(new)

        if not self.frontier:
            self.frontier.add(new)
            return True

        to_remove = set()
        for old in self.frontier:
            if self.dominates(old, new):
                # TODO delay return for even more error checking
                assert not to_remove, "Transitive property of dominance violated"
                return False
            if self.dominates(new, old):
                to_remove.add(old)

        for old in to_remove:
            self.frontier.remove(old)

        self.frontier.add(new)
        return True

    # TODO better name
    def would_add(self, new: T) -> bool:
        if not self.frontier:
            return True
        for old in self.frontier:
            if self.dominates(new, old):
                return True
        return False


def tuple_dominates(new: Tuple, old: Tuple) -> bool:
    assert len(new) == len(old)
    return any(n < o for n, o in zip(new, old)) and all(n <= o for n, o in zip(new, old))


def render_2d_frontier(frontier: ParetoFrontier[Tuple[float, float]], names: Tuple[str, str]):
    fig, ax = plt.subplots()

    all_times = []
    all_energies = []
    all_colors = []

    for t, e in frontier.all:
        all_times.append(t)
        all_energies.append(e)
        all_colors.append((t, e) in frontier.frontier)

    ax.scatter(x=all_times, y=all_energies, c=all_colors)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    fig.savefig("../ignored/schedules/frontier.png")
    plt.close(fig)
