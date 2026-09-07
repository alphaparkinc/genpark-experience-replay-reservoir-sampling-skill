"""
Experience Replay Reservoir Sampling Skill Client
Pure Python Standard Library implementation of streaming Experience Replay (Vitter's Algorithm R).
Maintains a bounded memory buffer of past task trajectories under unknown stream length,
guaranteeing uniform random distribution without memory bloat.
"""

from typing import List, Dict, Any, Tuple, Optional
import random


class ReservoirReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = 42):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.total_seen = 0
        self.rng = random.Random(seed)

    def add(self, item: Dict[str, Any]):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Random replacement with probability capacity / total_seen
            j = self.rng.randint(0, self.total_seen - 1)
            if j < self.capacity:
                self.buffer[j] = item

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        k = min(batch_size, len(self.buffer))
        return self.rng.sample(self.buffer, k)

    def get_task_distribution(self) -> Dict[str, int]:
        dist = {}
        for item in self.buffer:
            t = item.get("task_id", "unknown")
            dist[t] = dist.get(t, 0) + 1
        return dist
