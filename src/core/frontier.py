from typing import Tuple


class ParetoFrontier:
    """
    `progress` is maximized, `cost` is minimized.
    """

    def __init__(self, progress_size: int, cost_size: int):
        # TODO set or list for this? we really just want a better data structure
        self.best = set()
        self.progress_size = progress_size
        self.cost_size = cost_size

    def add(self, progress: Tuple, cost: Tuple) -> bool:
        """
        Try adding the given state to the frontier.
        Returns whether this state improves on the current frontier and was indeed added.
        """

        # TODO do we even need to split up progress and value? can we just have a single big frontier?
        # TODO how do we know which progresses are done and which aren't? should this data structure care about it?

        assert len(progress) == self.progress_size
        assert len(cost) == self.cost_size

        to_remove = set()

        for (bp, bc) in self.best:
            if progress <= bp and cost >= bc:
                return False
            if progress >= bp and cost <= bc:
                to_remove.add((bp, bc))
                # TODO can we break here?

        # TODO remove?
        assert len(to_remove) <= 1, "remove and document that this assumption is not actually right"
        for r in to_remove:
            self.best.remove(r)

        self.best.add((progress, cost))
        return True
