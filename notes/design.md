[//]: # (TODO move comments from the source code into this document to keep source readable)

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

* Try doing *nothing* first, certainly for channel operations and maybe even for core ops.
  * =immediately blacklist
* Add cuts (refuse to take branch if some condition is false)
* Add bounds (give up if we can't beat the current front)
  * Add conservative estimates to energy and timing to get higher bounds.  
* Symmetry breaking:
    * Intrinsic in the formulation: multiple actions that start at the same time should
      _not_ be tried in different orders.
    * Part of the neural network (eg. identical branch and join)
    * Part of the architecture (eg. uniform grid)
    * Switch to an event system where we only start operations if they could not have started earlier for some reason.
        * (eg. core occupied, no memory space (and couldn't drop yet because someone was using it), channel occupied, ...)

# Low level optimizations

* Early exit in dominance check
* Optimize data structures
* Support state undo actions
* Rewrite in Rust

# Extra features

* Constraint groups that prevent multiple object from running at the same time.
  Can be used to simulate a bunch of architectural features:
    * A core and channel that share the same memory port.
    * Multiple channels that form a single interconnect where both can't be active at the same time.
    * IMA core weight loading? The core running can block access to the dedicated RAM block.
      We could go even further and remove the concept of cores altogether, and say that mappings can just be part of
      constraint groups that represent the code, is this equivalent or even useful?
* Even more complex constraints, eg. full linear inequalities.
  * Would allow expressing peak power limitation, anything else?

* Allow recomputation of dropped values?