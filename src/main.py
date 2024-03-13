from core.problem import Hardware, Core, Memory, Channel, OperationGraph, OperationNode, Problem, OperationAllocation
from core.solver import schedule
import yappi

def main():
    # TODO experiment with multiple possible allocations with a latency/memory tradeoff

    # TODO find a case where splitting across two cores is optimal
    #   * get this working with a tiny accelerator first

    # hardware definition
    core_mem_size = 1024 * 1024

    bw_chip = 32
    bw_pcb = 32
    energy_chip = 1
    energy_pcb = 8

    offchip_memory = Memory("ram", None)
    memories = [Memory(f"mem-{i}", core_mem_size) for i in range(4)]
    cores = [Core(f"core-{i}", [memories[i]]) for i in range(4)]

    channels = [
        Channel("chan-01", memories[0], memories[1], True, True, 0, 1 / bw_chip, energy_chip),
        Channel("chan-23", memories[2], memories[3], True, True, 0, 1 / bw_chip, energy_chip),
        Channel("chan-02", memories[0], memories[2], True, True, 0, 1 / bw_chip, energy_chip),
        Channel("chan-13", memories[1], memories[3], True, True, 0, 1 / bw_chip, energy_chip),
        Channel("chan-ram", memories[0], offchip_memory, True, True, 0, 1 / bw_pcb, energy_pcb),
    ]
    hw = Hardware("hardware", cores, memories + [offchip_memory], channels)
    hw.assert_valid()
    hw.to_graphviz().render("../ignored/hardware", format="svg")

    # graph definition
    # TODO add weights (and maybe allow them to be stationary?)
    node_input = OperationNode("input", 1024, [])
    node_conv1 = OperationNode("conv1", 1024, [node_input])
    node_conv2a = OperationNode("conv2a", 1024, [node_conv1])
    node_conv2b = OperationNode("conv2b", 1024, [node_conv1])
    node_conv3a = OperationNode("conv3a", 1024, [node_conv2a])
    node_conv3b = OperationNode("conv3b", 1024, [node_conv2b])
    node_conv4 = OperationNode("conv4", 1024, [node_conv3a, node_conv3b])

    nodes = [node_input, node_conv1, node_conv2a, node_conv2b, node_conv3a, node_conv3b, node_conv4]
    inputs = [node_input]
    outputs = [node_conv4]
    graph = OperationGraph(id="graph", nodes=nodes, inputs=inputs, outputs=outputs)

    graph.assert_valid()
    graph.to_graphviz().render("../ignored/graph", format="svg")

    # problem definition
    allocations = {}
    for node in nodes:
        if node in inputs:
            continue
        allocations[node] = [
            OperationAllocation("basic", core, tuple([core.connected_memories[0]] * len(node.inputs)),
                                core.connected_memories[0], 10_000, 100)
            for core in cores
        ]
    placement_inputs = {n: offchip_memory for n in inputs}
    placement_outputs = {n: offchip_memory for n in outputs}
    problem = Problem(
        id="problem",
        hardware=hw,
        graph=graph,
        possible_allocations=allocations,
        placement_inputs=placement_inputs,
        placement_outputs=placement_outputs
    )
    problem.assert_valid()

    yappi.start()
    schedule(problem)
    yappi.stop()
    yappi.get_func_stats().print_all()


if __name__ == "__main__":
    main()
