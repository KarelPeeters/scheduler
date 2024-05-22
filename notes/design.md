[//]: # (TODO move comments from the source code into this document to keep source readable)

[//]: # (TODO steal remaining ideas from solver_event and delete it)

[//]: # (TODO turn this document into a checklist?)

# Core features

* Add partial starts:
  * NN layer that can start when only part of the previous output is available.
  * This should be enough to get streaming working properly!
  * Concretely:
    * For each input in an allocation, add a minimal fraction and a rate (assume simple and linear at first)
    * For each allocation output, define a similar stream curve.
    * For channel operations, allow streaming at lower bandwidth.
  * Hopefully this is enough that we can avoid splitting layers up, causing combinatorial explosions.

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

* Switch to proper graph representation
  * Multiple outputs for a node
  * Maybe even remove the graph thing entirely, and switch to only using mappings?
  * Slice, transpose, concat, ...
    * Slice probably just implicit,
    * For transpose just assume that copy engines are good enough that we can ignore it?
    * Concat just adds memory allocation constraints.

# Current issues

The biggest issues right now are:

* runtime for a single linear network is O(2^N), we need it to be polynomial at worst
  * needs some extra pruning trick, eg. only allow doing "no action" if there's some advantage to that later
* runtime for branches (even just 1xN) is horrible
  * needs more investigation, probably better symmetry breaking?
* runtime for networks with weights is bad, there's way too much symmetry in copying them
  * probably solved by better symmetry breaking?

# High level optimizations

* Try different frontier data structure:
  * each node has lower and upper bound for _each_ index, allowing even more skipping?
  * node is a state that points to other states that dominate it (except for time and energy)
    * hopefully this allows for some efficient tree walking thing for checking existence?
  * find an ever more different structure where states can be compared more directly
    * eg. dropping as-of-yet useless actions from an existing state should immediately be done when comparing
    * alternative formulation: when adding a state all partial states should immediately be "added" too
* Expand symmetry breaking to:
  * allow time differences
  * allow pushing around copies if possible
* Try removing useless actions instead of pruning, this hopefully fills the frontiers faster with useful states.
  * Maybe go even further and back-edit old partial states?
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
* Gradually expand the solution space to try to get full schedules as quickly as possible, enabling more cuts.
  * This can take many forms, eg. limit the allocations that can be used, the cores, ...
* Cut for channels that connect the same segments and are dominated:
  * Worse channel can only be used if during its runtime the better one was also used at some point.
* Progressive widening: start with low bounds for time and energy and gradually increase,
  hoping to find the best solution first.
* Sort queue by time and energy per finished node.

# Input preprocessing
* Drop dominated allocations (and maybe channels if that's possible)?
* Remove impossible to use channels, memories, allocations.
* Sort allocations by lower energy and time, hopefully this better deals with the combinatorial
  triangle explosion when there are multiple choices (eg. eff, mid, fast).
  * Plot some graphs to see how bad it currently is, are we doing O(N^2)?

# Low level optimizations

* Optimize the pareto checks
  * Keywords: skyline data structure, R-trees
  * Can we construct an R-tree incrementally?
  * Do we even need a full R-tree? Or is some cursed hashmap structure based on index values enough?
  * Resources:
    * https://github.com/phill-holland/kdtree-pareto-front
    * http://www.cse.cuhk.edu.hk/~taoyf/course/infs4205/lec/skyline.pdf
    * https://www.sciencedirect.com/science/article/abs/pii/S0020025519306176
    * https://static.aminer.org/pdf/PDF/000/211/201/on_the_computational_complexity_of_finding_the_maxima_of_a.pdf
    * https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=6430ffb349e49560b9beced5d413216953c0ab35
    * https://www.mdpi.com/2076-3417/10/19/6858
    * https://www.comp.nus.edu.sg/~atung/publication/k_dominant.pdf
    * https://users-cs.au.dk/gerth/papers/icalp11.pdf
    * https://cs.stackexchange.com/questions/141930/children-of-internal-node-in-a-quadtree-with-high-dimensionality
      * https://tzaeschke.github.io/phtree-site/
  * some rough stats: frontier.add adds a new state 70% of the time, and it only ever really deletes <2 states
  * Switch to storing compact state keys instead of full state instances.
    * Eg. a sorted index->float vector representing an index map? Or a compact hashmap implementation.

* Early exit in dominance check
* Optimize data structures
  * Frontier cache (with a dynamically scaling size depending on hit rates?)
  * For starting operations: immediately iterate only over actions that have actually been triggered?
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

* Allow actions that take zero time, energy and only deal with values of size 0.
  * These might be used to encode extra runtime dependencies?

# Test cases
* simple network
* disastrous highly symmetric hardware and network
* real networks!
* tricky memory cases, eg. compute something, temporarily move it to a separate memory, drop it, then copy it back 