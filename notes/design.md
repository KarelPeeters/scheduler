[//]: # (TODO move comments from the source code into this document to keep source readable)

[//]: # (TODO steal remaining ideas from solver_event and delete it)

[//]: # (TODO turn this document into a checklist?)

# Core features

* Improve pareto dominance check:
    * include core active times?
    * include value availability
    * Make sure this is completely bug-free, pessimistic if necessary!

* Add value dropping.
    * Don't allow dropping values that have not yet been used at least once.
    * Don't allow dropping the last value.
    * Only drop values if there's actually an operation that will need the extra memory space.

* Check that execution is _possible_ first using some dumb handcoded algorithm?
    * Prevents looping forever on illegal situations.
    * Immediately gets a bound so we can start pruning.

# High level optimizations

* Check for _graph automorphisms_ during dominance check.
  * Precompute them for both the HW and NN in advance, then hopefully it's cheap enough to evaluate.
  * Also look into _graph canonization_, maybe that's even better. 
* Try doing *nothing* first, certainly for channel operations and maybe even for core ops.
  * =immediately blacklist
* Add cuts (refuse to take branch if some condition is false)
  * Don't allow more instances of a value to be alive (and not-used) then will be used in the future.
* Add bounds (give up if we can't beat the current front)
  * Add conservative estimates to energy and timing to get higher bounds.
    * Improve these: eg. for min time consider currently occupied cores and a real mini schedule.
* Symmetry breaking:
    * Intrinsic in the formulation: multiple actions that start at the same time should
      _not_ be tried in different orders.
    * Part of the neural network (eg. identical branch and join)
    * Part of the architecture (eg. uniform grid)
    * Switch to an event system where we only start operations if they could not have started earlier for some reason.
        * (eg. core occupied, no memory space (and couldn't drop yet because someone was using it), channel occupied, ...)
* Improve pareto check
  * Mark earlier availability of values better?
* Post-pruning: prune useless operations once we reach a done state, immediately setting better bounds.
  * We can prune even earlier, eg. as soon as a value is dead we can prune all copies of it that haven't been used.
* Try shuffling the order of actions instead of picking some potentially worst case fixed ordering.
* Drop useless operations in done state to get tighter bound.
  * ie. still-running and transfers that didn't get used
* Don't take actions that could have been taken earlier.

# Input preprocessing
* Drop dominated allocations (and maybe channels if that's possible)?
* Remove impossible to use channels, memories, allocations.

# Low level optimizations

* Early exit in dominance check
* Optimize data structures
  * Frontier cache (with a dynamically scaling size depending on hit rates?)
  * For starting operations: immediatly iterate only over actions that have actually been triggered?
* Support state undo actions
* Rewrite in Rust
  * Bitsets for sets where possible.
* Merge both the done and partial frontier?
  * This would probably involve integrating the estimates better?
* Write all checks so condition checks fail as soon as possible.

# Extra features

* Constraint groups that prevent multiple object from running at the same time.
  Can be used to simulate a bunch of architectural features:
    * We can get rid of channel directionality.
    * A core and channel that share the same memory port.
    * Multiple channels that form a single interconnect where both can't be active at the same time.
    * IMA core weight loading? The core running can block access to the dedicated RAM block.
      We could go even further and remove the concept of cores altogether, and say that mappings can just be part of
      constraint groups that represent the code, is this equivalent or even useful?
* Even more complex constraints, eg. full linear inequalities.
  * Would allow expressing peak power limitation, anything else?

* Allow recomputation of dropped values?

* Allow cyclic requirement: certain values must be _somewhere_, but allocation is free to choose where.
* Also allow multiple inputs and outputs places for a single value.

# Test cases
* simple network
* disastrous highly symmetric hardware and network
* real networks!
* tricky memory cases, eg. compute something, temporarily move it to a separate memory, drop it, then copy it back 