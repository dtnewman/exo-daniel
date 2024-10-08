from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition

class HybridPartitioningStrategy(PartitioningStrategy):
    def __init__(self, flops_weight: float = None, memory_weight: float = None):


        self.flops_weight = flops_weight or (1.00 - memory_weight) if memory_weight is not None else 0.5
        self.memory_weight = memory_weight or (1.00 - self.flops_weight)

        if abs(self.flops_weight + self.memory_weight - 1.00) > 1e-5:
            raise ValueError("flops_weight and memory_weight must add up to 1.00")

    def partition(self, topology: Topology) -> List[Partition]:
        """
        This strategy gives weights to nodes based on their flops and memory,
        normalizing them to ensure the scales are comparable.
        """
        nodes = list(topology.all_nodes())

        # Get max FLOPS and memory to normalize the values
        max_flops = max(node[1].flops.fp32 for node in nodes)
        max_memory = max(node[1].memory for node in nodes)

        # Avoid division by zero in case all flops or memory are 0
        max_flops = max_flops if max_flops > 0 else 1
        max_memory = max_memory if max_memory > 0 else 1

        # Give each node a normalized weight based on flops and memory
        for node in nodes:
            normalized_flops = node[1].flops.fp32 / max_flops
            normalized_memory = node[1].memory / max_memory
            node[1].weight = normalized_flops * self.flops_weight + normalized_memory * self.memory_weight

        # Sort by weight in descending order, then by node ID
        nodes.sort(key=lambda x: (-float(x[1].weight), x[0]))

        total_weight = sum(node[1].weight for node in nodes)

        partitions = []
        start = 0
        for i, node in enumerate(nodes):
            is_last_node = i == len(nodes) - 1
            end = 1.0 if is_last_node else round(start + (node[1].weight / total_weight), 5)
            partitions.append(Partition(node[0], start, end))
            start = end

        return partitions
