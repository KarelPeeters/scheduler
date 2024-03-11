# Scheduler

## Initial idea

How far can we get with a basic branch-and-bound solver for NN scheduling on complex multicore accelerators?

Steps:

* first get the basics working with just summed memory usage, basic cores, unsplittable operations, basic memory transfer links, ...
    * hopefully this is relatively easy to implement and finds the full Pareto in a reasonable time
* add in real ILP-based memory allocation
    * enable full start/end condition requirements
* gradually add more complications:
    * memory links where only one pair can be active at a time (eg. because of a shared interconnect or bus)
    * real memory hierarchies where operations can be split up
        * to be determined: is this in-scope or do we leave this for specific loop tiling schedulers?

## Sources of inspiration

* Stream: https://kuleuven-micas.github.io/stream/, https://github.com/kuleuven-micas/stream
* This specific question from a programming competition: https://github.com/vlaamseprogrammeerwedstrijd/opgaves/blob/master/2024/cat4/tevreden/tevreden.pdf
* The AlphaZero papers.


## New ideas

* can we formulate this as a big ILP instead?
    * sadly these get only one solution, so we'd have to run it multiple times to get the pareto front
    * can these even deal with pathfinding-style mem-copies?
