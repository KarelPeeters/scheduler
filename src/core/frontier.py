from typing import TypeVar, Generic, Callable, Set

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
        assert len(to_remove) <= 1, "remove and document that this assumption is not actually right"
        for old in to_remove:
            self.frontier.remove(old)

        self.frontier.add(new)
        return True
