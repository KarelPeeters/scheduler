from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from core.action import Action, ActionCore, ActionChannel, ActionWait
from core.problem import Memory, Problem


# TODO add separately written check_valid that ensures the schedule is valid?
#   mostly as a check that the solver didn't hit a bug
@dataclass(frozen=True, eq=False)
class Schedule:
    problem: Problem
    actions: List[Action]

    @property
    def time_bounds(self):
        start = min((a.time_start for a in self.actions), default=0)
        end = max((a.time_end for a in self.actions), default=0)
        return start, end

    def plot_all(self):
        fig, [ax0, ax1] = plt.subplots(2)
        self.plot_schedule_actions(ax0)
        self.plot_energy(ax1, spread=False)
        # self.plot_energy(ax2, spread=True)
        self.plot_memories(limits=False)
        plt.show(block=True)

    def plot_energy(self, ax, spread: bool):
        if spread:
            # TODO this is pretty inefficient
            values_time = sorted(t for a in self.actions for t in [a.time_start, a.time_end])
            values_energy = []

            for time in values_time:
                energy = 0
                for action in self.actions:
                    t = (time - action.time_start) / (action.time_end - action.time_start)
                    if 0 <= t:
                        energy += action.energy * min(t, 1)
                values_energy.append(energy)
        else:
            values_time = [0]
            values_energy = [0]

            curr_energy = 0
            for action in self.actions:
                next_energy = curr_energy + action.energy

                values_time.extend([action.time_start, action.time_start])
                values_energy.extend([curr_energy, next_energy])

                curr_energy = next_energy

            values_time.append(self.time_bounds[1])
            values_energy.append(curr_energy)

        print("time", values_time)
        print("energy", values_energy)

        ax.plot(values_time, values_energy)
        ax.set_xlabel("Time")
        kind_str = "spread" if spread else "instant"
        ax.set_ylabel(f"Energy ({kind_str})")
        ax.set_xlim(*self.time_bounds)

    def plot_memories(self, limits: bool):
        hw = self.problem.hardware

        # collect data
        values_time_bits = {mem: ([0.0], [0]) for mem in hw.memories}
        for value, mem in self.problem.placement_inputs.items():
            values_time_bits[mem][1][0] += value.size_bits

        def add_bits(mem: Memory, time: float, bits: int):
            pair = values_time_bits[mem]
            filling_before = pair[1][-1]
            filling_after = filling_before + bits
            pair[0].extend([time, time])
            pair[1].extend([filling_before, filling_after])

        for action in self.actions:
            if isinstance(action, ActionWait):
                # wait actions have no memory implications
                pass
            elif isinstance(action, ActionCore):
                add_bits(mem=action.alloc.output_memory, time=action.time_start, bits=action.node.size_bits)
            elif isinstance(action, ActionChannel):
                add_bits(mem=action.dest, time=action.time_start, bits=action.value.size_bits)
            else:
                raise ValueError(f"Unknown action type: {action}")

        # add final points
        time_start, time_end = self.time_bounds
        for mem in hw.memories:
            add_bits(mem, time_end, 0)

        # plotting itself
        fig, axes = plt.subplots(len(hw.memories), squeeze=False, sharex="all")
        axes = np.squeeze(axes, 1)

        for mem_index, mem in enumerate(hw.memories):
            ax = axes[mem_index]
            ax.plot(*values_time_bits[mem])

            if limits and mem.size_bits is not None:
                ax.axhline(mem.size_bits, color='k', linestyle='dashed')

            ax.set_ylabel(f"{mem.id}\noccupancy in bits")
            ax.set_xlabel("Time")

        return fig

    def plot_schedule_actions(self, ax):
        hw = self.problem.hardware
        actions = self.actions

        for action in actions:
            time_mid = (action.time_start + action.time_end) / 2

            if isinstance(action, ActionWait):
                # wait actions are not explicitly visualized
                pass
            elif isinstance(action, ActionCore):
                core_index = hw.cores.index(action.alloc.core)

                rect_xy = (action.time_start, core_index - 0.5)
                rect = plt.Rectangle(rect_xy, action.alloc.time, 1, color='green', fill=False)
                ax.add_patch(rect)

                ax.text(time_mid, core_index, f"{action.node.id}\n({action.alloc.id})", ha='center', va='center')
            elif isinstance(action, ActionChannel):
                chan_index = hw.channels.index(action.channel)

                rect_xy = (action.time_start, len(hw.cores) + chan_index - 0.5)
                rect = plt.Rectangle(rect_xy, action.total_latency, 1, color='orange', fill=False)
                ax.add_patch(rect)

                if action.source == action.channel.memory_a:
                    dir = "->"
                elif action.source == action.channel.memory_b:
                    dir = "<-"
                else:
                    dir = "?"

                ax.text(time_mid, len(hw.cores) + chan_index, f"{action.value.id}\n{dir}", ha='center', va='center')
            else:
                raise ValueError(f"Unknown action type: {action}")

        # update bounds (pyplot doesn't do this automatically for patches)
        time_start, time_end = self.time_bounds
        if time_end > time_start:
            ax.set_xlim(time_start, time_end)
        ax.set_ylim(-0.5, len(hw.cores) + len(hw.channels) - 0.5)

        ax.invert_yaxis()
        ax.set_yticks(range(len(hw.cores) + len(hw.channels)))
        ax.set_yticklabels([core.id for core in hw.cores] + [chan.id for chan in hw.channels])
