from typing import Tuple, Generic


class ParetoFrontier:
    """ Stores a pareto frontier of _maximized_ items. """

    def __init__(self, item_size: int):
        self.frontier = set()
        self.item_size = item_size

    def add(self, new: Tuple) -> bool:
        """
        Try adding the given state to the frontier.
        Returns whether this state improves on the current frontier and was indeed added.
        """

        assert isinstance(new, tuple)
        assert len(new) == self.item_size

        to_remove = set()

        for old in self.frontier:
            if all(a <= b for a, b in zip(new, old)):
                # new is dominated by old
                return False

            if all(a > b for a, b in zip(new, old)):
                # new dominates old
                to_remove.add(old)
                # TODO can we break here?

        # TODO remove?
        assert len(to_remove) <= 1, "remove and document that this assumption is not actually right"
        for r in to_remove:
            self.frontier.remove(r)

        self.frontier.add(new)
        return True
