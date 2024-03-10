# AlphaZero for neural network scheduling

## Initial idea

I wonder if A0 would do well at the problem of scheduling a neural network on an accelerator that consists of multiple
blocks of memory and multiple sub-accelerators, connected in some "arbitrary graph" way. This is becoming a very
relevant problem with the recent AI accelerator and compiler demand!

The typical approaches so far are:

* manually do it (too much work)
* manually write some compiler heuristics and hope it works (suboptimal, I think this is what most current compilers are
  doing)
* put everything into a big ILP and add some heuristics to get it fast enough,
  eg. https://arxiv.org/pdf/2311.18246.pdf (doesn't scale to complex accelerators and memory hierarchies)
* try to put it in a evolutionary algorithm (eg. https://github.com/KULeuven-MICAS/stream, what I was using for my
  thesis. Does not work very well despite a couple of papers already having been written about it)

I wonder how well A0 would do at this. It's a sequential decision process after all, you just need to choose which
operation to run on which accelerator core at each time step, sequentially in time.

The big unknowns are:

* Does A0 work well in these single player setups? You lose the stabilization you get from adversarialness.
* Is it ever going to be fast enough? Hopefully it is, there's the training cost but then for "tournaments" you just
  need to do one big "game" from front to back to get the final schedule.
* Is it possible to come up with some "selfplay" dataset of neural networks and accelerator designs that's varied enough
  that you can just do one big training run and end up with a NN that's general enough to schedule anything on anything?
* What to do about memory allocations? Calling out to an ILP solver is probably best, but when? After each time step?
  Only at the end of the "game"/schedule and adding this as an auxiliary head? More generally what exactly is the value
  head going to predict? "%success given this time and memory constraint"? "expected time, expected memory"?
* What NN arch and IO design is good for this? Both accelerators and NNs are graphs, but I get the impression that GNNs
  don't really work all that well. Does a big transformer with a simple input encoding just work™?

I've found one paper with the keywords "alphazero
scheduling": https://www.researchgate.net/publication/341204115_Sheet-Metal_Production_Scheduling_Using_AlphaGo_Zero.
It's not NN scheduling ofc, and the paper seems kind of amateurish, but it's something.

## Problem formulation

Given:

* A neural network. In general this is some arbitrary graph of tensors and operations between them. The operations are
  eg. a convolution, or smaller pieces, eg. a convolution but only the first N batch elements or channels.
* Some accelerator architecture. Which in general is also some arbitrary graph consisting of
    * accelerator cores (eg. a CPU, a matrix multiply unit, systolic array, an analog compute core, an nvidia streaming
      multiprocessor, ...)
        * these have limitations, eg. can only do matmul and no conv, can't do activations functions, ...
        * these also have power usages and latencies per supported operation
    * memory blocks (eg. L1, L2, L3, DRAM from typical CPUs, or more exotic arrangements, eg. shared memory in nvidia
      GPUs, manually managed local memory buffers from TPUs or Tesla Dojo type stuff, ...)
        * the limitation here is mostly size
    * data transfer busses between cores and memory blocks or between different memory blocks. The DDR5 bus, in-chip
      interconnects, CPU cache coherency busses, nvidia infinity fabric, network-on-chip stuff, ...)
        * these have latencies, throughputs and power usage per bit of information transferred

The problem is: How do we map the neural network onto the hardware, minimizing any one or some combination of:

* total energy used
* total latency
* peak memory usage

The decisions that need to be made are:

* which operation runs on which core
* what order do operations run in
* when do memory transfers between the different blocks happen

I'm thinking of formulating this as a game/sequential decision process as follows:

* The accelerator and the NN are just static throughout the game, kind of like how the handicap is a fixed setting in
  Go.
* The states are partial schedules.
* The starting state is "nothing has been scheduled yet".
* The available moves are:
    * if any operation/transfer is currently running, wait for the earliest one to finish
    * for any unscheduled operation for each idle core that can run it: schedule it on that core
    * for any unscheduled transfer between memory blocks for each idle memory bus that can run it: start the transfer on
      that bus
* The end state is when the network has been fully scheduled.
* The value function is just whatever the chosen metric to optimize is.

It feels like all of this should work well with AlphaZero! 
