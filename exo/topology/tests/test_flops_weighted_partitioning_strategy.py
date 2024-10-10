import unittest
from exo.topology.flops_weighted_partitioning_strategy import FlopsWeightedPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition


class TestFlopsWeightedPartitioningStrategy(unittest.TestCase):
  def test_partition(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    topology = Topology()
    topology.update_node(
      "node1",
      DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=4.0, fp16=8.0, int8=16.0), avg_processing_time=0, weight=1),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0), avg_processing_time=0, weight=1),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(model="test3", chip="test3", memory=6000, flops=DeviceFlops(fp32=10.0, fp16=20.0, int8=40.0), avg_processing_time=0, weight=1),
    )
    topology.add_edge("node1", "node2")
    topology.add_edge("node2", "node3")
    topology.add_edge("node3", "node1")
    topology.add_edge("node1", "node3")

    strategy = FlopsWeightedPartitioningStrategy()
    partitions = strategy.partition(topology)

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node3", 0.0, 0.66667),
        Partition("node1", 0.66667, 0.93334),
        Partition("node2", 0.93334, 1.0),
      ],
    )

  def test_partition_rounding(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
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
        memory=192*1024*1024*1024,
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

    strategy = FlopsWeightedPartitioningStrategy()
    partitions = strategy.partition(topology)
    
    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node2", 0.0, 0.66667),
        Partition("node1", 0.66667, 0.83334),
        Partition("node3", 0.83334, 1.0),
      ],
    )

  def test_partition_with_no_flops(self):
    # two nodes with no flops

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
        memory=128*1024*1024*1024,
        flops=DeviceFlops(fp32=0.0, fp16=0.0, int8=0.0),
      ),
    )

    strategy = FlopsWeightedPartitioningStrategy()
    # should fall back to ring-memory-weighted partitioning
    partitions = strategy.partition(topology)
    self.assertEqual(len(partitions), 2)
    self.assertEqual(
      partitions,
      [
        Partition("node1", 0.0, 0.5),
        Partition("node2", 0.5, 1.0)
      ],
    )


if __name__ == "__main__":
  unittest.main()
