import unittest
from exo.topology.hybrid_partitioning_strategy import HybridPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition


class TestHybridPartitioningStrategy(unittest.TestCase):
    def test_partition(self):
        # two nodes
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=8000, flops=DeviceFlops(fp32=2.5, fp16=5.0, int8=10.0)),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(model="test2", chip="test2", memory=16000, flops=DeviceFlops(fp32=12.5, fp16=25.0, int8=50.0)),
        )

        topology.add_edge("node1", "node2")
        topology.add_edge("node2", "node1")

        # Use a 50-50 split between flops and memory
        strategy = HybridPartitioningStrategy(flops_weight=0.5, memory_weight=0.5)
        partitions = strategy.partition(topology)

        self.assertEqual(len(partitions), 2)

        # max_flops = 12.5
        # max_memory = 16000

        # normalized_flops1 = 2.5 / 12.5 = 0.2
        # normalized_memory1 = 8000 / 16000 = 0.5
        # weight1 = 0.2 * 0.5 + 0.5 * 0.5 = 0.35


        # normalized_flops2 = 12.5 / 12.5 = 1.0
        # normalized_memory2 = 16000 / 16000 = 1.0
        # weight2 = 1.0 * 0.5 + 1.0 * 0.5 = 1.0
        self.assertEqual(
            partitions,
            [
                Partition("node2", 0.0, 0.74074), # 1 / (1.0 + 0.35)
                Partition("node1", 0.74074, 1.0),
            ],
        )

    def test_partition_two_equals(self):
        # three nodes
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test1",
                memory=128*1024*1024*1024,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(
                model="Mac Studio",
                chip="test2",
                memory=256*1024*1024*1024,
                flops=DeviceFlops(fp32=4.0, fp16=8.0, int8=10.0),
            ),
        )
        topology.update_node(
            "node3",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test3",
                memory=128*1024*1024*1024,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )

        strategy = HybridPartitioningStrategy(flops_weight=0.7, memory_weight=0.3)
        partitions = strategy.partition(topology)
        
        self.assertEqual(len(partitions), 3)
        self.assertEqual(
            partitions,
            [
                Partition("node2", 0.0, 0.60606), 
                Partition("node1", 0.60606, 0.80303),
                Partition("node3", 0.80303, 1.0),
            ],
        )

    def test_partition_with_no_flops(self):
        # three nodes

        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test1",
                memory=128*1024*1024*1024,
                flops=DeviceFlops(fp32=0.0, fp16=0.0, int8=0.0),
            ),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(
                model="Mac Studio",
                chip="test2",
                memory=256*1024*1024*1024,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )
        topology.update_node(
            "node3",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test3",
                memory=256*1024*1024*1024,
                flops=DeviceFlops(fp32=0.0, fp16=0.0, int8=0.0),
            ),
        )

        # Use a strategy that weights memory only
        strategy = HybridPartitioningStrategy(flops_weight=0.0, memory_weight=1.0)
        partitions = strategy.partition(topology)

        self.maxDiff = None
        self.assertEqual(len(partitions), 3)
        self.assertEqual(
            partitions,
            [
                Partition("node2", 0.0, 0.4),
                Partition("node3", 0.4, 0.8),
                Partition("node1", 0.8, 1.0),
            ],
        )

    def test_partition_with_flops_only(self):
        # A bit contrived, but we want to make sure we don't divide by zero

        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test1",
                memory=0.0,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(
                model="Mac Studio",
                chip="test2",
                memory=0.0,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )
        topology.update_node(
            "node3",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test3",
                memory=0.0,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )
        
        # Use a strategy that weights flops only    
        strategy = HybridPartitioningStrategy(flops_weight=1.0, memory_weight=0.0)
        partitions = strategy.partition(topology)

        self.assertEqual(len(partitions), 3)
        self.assertEqual(
            partitions,
            [
                Partition("node1", 0.0, 0.33333),
                Partition("node2", 0.33333, 0.66666),
                Partition("node3", 0.66666, 1.0),
            ],
        )

    def test_partition_with_memory_only(self):
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test1",
                memory=128*1024*1024*1024,
                flops=DeviceFlops(fp32=0.0, fp16=0.0, int8=0.0),
            ),
        )   
        topology.update_node(
            "node2",
            DeviceCapabilities(
                model="Mac Studio",
                chip="test2",
                memory=256*1024*1024*1024,
                flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0),
            ),
        )   
        topology.update_node(
            "node3",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test3",
                memory=256*1024*1024*1024,
                flops=DeviceFlops(fp32=0.0, fp16=0.0, int8=0.0),
            ),
        ) 

        # Use a strategy that weights memory only
        strategy = HybridPartitioningStrategy(flops_weight=0.0, memory_weight=1.0)
        partitions = strategy.partition(topology)

        self.assertEqual(len(partitions), 3)
        self.assertEqual(
            partitions,
            [
                Partition("node2", 0.0, 0.4),
                Partition("node3", 0.4, 0.8),
                Partition("node1", 0.8, 1.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
