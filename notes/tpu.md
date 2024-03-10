# Notes on the TPU hardware architecture

Based on "The Design Process for Google's Training Chips: TPUv2 and TPUv3"
available at https://ieeexplore.ieee.org/document/9351692

Memory sizes:
* HBM: doubled since previous gen

Bandwidths:
* HBM: 700 GB/s/chip
* four external links: 500 Gb/s = 60 GB/s
* two internal links: twice as fast = 120 GB/s
* PCIe: 16 GB/s/chip
* in-package bandwidth: 700 GB/s/chip (what does this mean?)

Changes for V3:
* doubling matmul units (larger units or more units?)
* +30% HBM bandwidth
* double HBM size (to what?)
