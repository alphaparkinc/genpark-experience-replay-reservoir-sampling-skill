"""
Demonstration of Experience Replay Reservoir Sampling Skill
"""

from client import ReservoirReplayBuffer

def main():
    print("=== Streaming Experience Replay with Reservoir Sampling ===")
    buffer = ReservoirReplayBuffer(capacity=10)

    # Stream experiences from 3 sequential tasks
    print("Streaming 100 experiences across Tasks A, B, C...")
    for i in range(100):
        task_label = "task_A" if i < 30 else ("task_B" if i < 70 else "task_C")
        buffer.add({"step": i, "task_id": task_label, "reward": 1.0 / (i + 1)})

    dist = buffer.get_task_distribution()
    print(f"Total Streamed: {buffer.total_seen}, Buffer Size: {len(buffer.buffer)}")
    print("Buffer Task Distribution:", dist)

    batch = buffer.sample_batch(4)
    print(f"Sampled Replay Batch of size {len(batch)}:")
    for b in batch:
        print(f"  Step: {b['step']:02d} | Task: {b['task_id']}")

    assert len(buffer.buffer) == 10
    assert len(dist) >= 2
    print("Experience Replay Reservoir Sampling Verification PASS!")

if __name__ == "__main__":
    main()
