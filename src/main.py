from core.problem import Hardware, Core, Memory, Channel


def main():
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
    hw = Hardware("grid", cores, memories + [offchip_memory], channels)
    hw.assert_valid()
    hw.to_graphviz().render("../ignored/hardware", format="svg")


if __name__ == "__main__":
    main()
