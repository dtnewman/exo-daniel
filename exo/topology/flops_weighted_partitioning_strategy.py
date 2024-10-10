from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition
from .ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy

class FlopsWeightedPartitioningStrategy(PartitioningStrategy):
  def partition(self, topology: Topology) -> List[Partition]:
    """
    This strategy partitions the topology by flops, giving priority to nodes
    with the highest flops. Assumes that at least one node has flops. No 
    allocation is made for nodes with no flops.
    """
    nodes = list(filter(lambda x: x[1].flops.fp32 > 0, topology.all_nodes()))

    if len(nodes) == 0:
      print("No flops found in the topology, defaulting to memory-weighted partitioning")
      return RingMemoryWeightedPartitioningStrategy().partition(topology)

    # Sort by flops in descending order, then memory in descending order, then by node ID
    nodes.sort(key=lambda x: (-float(x[1].flops.fp32), -float(x[1].memory), x[0]))

    total_flops = sum(node[1].flops.fp32 for node in nodes)
    
    partitions = []
    start = 0
    for i, node in enumerate(nodes):
      is_last_node = i == len(nodes) - 1
      end = 1.0 if is_last_node else round(start + (node[1].flops.fp32/total_flops), 5)
      partitions.append(Partition(node[0], start, end))
      start = end
      
    return partitions
