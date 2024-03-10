from core.problem import Hardware, Core, Memory, Channel, OperationGraph, OperationNode


def main():
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
    node_input = OperationNode("input", 1024, [])
    node_conv1 = OperationNode("conv1", 1024, [node_input])
    node_conv2 = OperationNode("conv2", 1024, [node_conv1])
    node_conv3 = OperationNode("conv3", 1024, [node_conv2])
    node_conv4 = OperationNode("conv4", 1024, [node_conv3])

    nodes = [node_input, node_conv1 , node_conv2, node_conv3, node_conv4]
    inputs = [node_input]
    outputs = [node_conv4]
    graph = OperationGraph(id="graph", nodes=nodes, inputs=inputs, outputs=outputs)

    graph.assert_valid()
    graph.to_graphviz().render("../ignored/graph", format="svg")

    # problem definition


if __name__ == "__main__":
    main()
