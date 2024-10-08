from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    # nodes.sort(key=lambda x: (x[1].memory, x[0]))  # Sort by memory in ascending order, then by node ID
    # 15.39/3.3/154
    # 8.92/5.6/330
    nodes.sort(key=lambda x: (-x[1].memory, x[0]))  # Sort by memory in descending order, then by node ID

    print('nodes:')
    for node in nodes:
      print(f"{node[0]}: {node[1].memory}")

    total_memory = sum(node[1].memory for node in nodes)
    partitions = []
    start = 0
    for i, node in enumerate(nodes):
      end = 1.0 if i == len(nodes) - 1 else round(start + (node[1].memory/total_memory), 5)
      partitions.append(Partition(node[0], start, end))
      start = end
    return partitions
