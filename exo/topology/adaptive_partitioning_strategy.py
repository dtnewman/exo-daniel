from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition

class AdaptivePartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        """
        Experimental: This strategy uses weights to determine ordering and partitioning of nodes.
        The weights are assigned an initial value of 1 on the first API request. They then get
        updated stochastically based on the processing times of the requests. If a subsequent
        request is processed faster than the previous one, the weights remain the same. If a
        subsequent request is processed slower, the weights are adjusted back to where they were
        before.
        """
        nodes = list(topology.all_nodes())
        nodes.sort(key=lambda x: (-x[1].weight, -x[1].memory, x[0]))  # Sort by weight and then memory in descending order, then by node ID
        total_weight = sum(node[1].weight for node in nodes)
        partitions = []
        start = 0
        for i, node in enumerate(nodes):
            end = 1.0 if i == len(nodes) - 1 else round(start + (node[1].weight/total_weight), 5)
        partitions.append(Partition(node[0], start, end))
        start = end
        return partitions