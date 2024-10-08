from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class FlopsWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    nodes.sort(key=lambda x: (-float(x[1].flops.fp32), x[0]))  # Sort by flops in descending order, then by node ID

    total_flops = sum(node[1].flops.fp32 for node in nodes)
    partitions = []
    start = 0
    for i, node in enumerate(nodes):
      is_last_node = i == len(nodes) - 1
      end = 1.0 if is_last_node else round(start + (node[1].flops.fp32/total_flops), 5)
      partitions.append(Partition(node[0], start, end))
      start = end
    return partitions
