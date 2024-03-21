from typing import TypeVar, Generic, Callable, Set, Tuple

import matplotlib.pyplot as plt
import pylru

T = TypeVar("T")


class ParetoFrontier(Generic[T]):
    """ Stores a pareto frontier of _maximized_ items. """

    def __init__(self, dominates: Callable[[T, T], bool]):
        self.frontier: Set[T] = set()

        # TODO figure out optimal cache size
        self.frontier_cache: pylru.lrucache = pylru.lrucache(256)
        self.frontier_hits = 0
        self.cache_hits = 0
        self.add_attempts = 0

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
            self.frontier_cache[new] = None
            return True

        # TODO multiple layers of caches of increasing size that each get the spillover from the next one?
        #   or is that equivalent to just iterating over a single frontier in-order of recency?
        #   think about memory locality!
        if self.add_attempts != 0 and self.add_attempts % 1000 == 0:
            print(f"Cache hit rate: {self.cache_hits / self.add_attempts}, frontier hit rate: {self.frontier_hits / self.add_attempts}")
            self.cache_hits = 0
            self.add_attempts = 0

        self.add_attempts += 1
        for old in self.frontier_cache.keys():
            if self.dominates(old, new):
                # re-insert to keep alive
                self.cache_hits += 1
                self.frontier_cache[new] = None
                return False

        to_remove = set()
        for old in self.frontier:
            if self.dominates(old, new):
                # TODO delay return for even more error checking
                assert not to_remove, "Transitive property of dominance violated"

                # add to cache
                self.frontier_cache[old] = None
                self.frontier_hits += 1

                return False
            if self.dominates(new, old):
                to_remove.add(old)

        for old in to_remove:
            self.frontier.remove(old)
            # TODO is dropping old values useful?
            if old in self.frontier_cache:
                del self.frontier_cache[old]

        self.frontier.add(new)
        # TODO is adding new items immediately useful?
        self.frontier_cache[new] = None
        return True

    # TODO better name
    def would_add(self, new: T) -> bool:
        if not self.frontier:
            return True
        for old in self.frontier:
            if self.dominates(old, new):
                return False
        return True


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
    fig.savefig("../../ignored/schedules/frontier.png")
    plt.close(fig)
