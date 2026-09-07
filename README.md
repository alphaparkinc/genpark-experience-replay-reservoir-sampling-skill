# GenPark Experience Replay Reservoir Sampling Skill

Continual learning streaming experience replay buffer with Vitter's reservoir sampling.

Discover more agent tools at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph LR
    A[Continuous Agent Trajectories] -->|P = C / N| B[Reservoir Sampling Buffer]
    B -->|Uniform Sample Batch| C[Continual Replay Optimizer]
    C --> D[Interleaved Experience Gradient Updates]
```

## Features
- Bounded $O(C)$ capacity memory footprint.
- Uniform probability guarantee across arbitrarily long streams.
- Zero external dependencies.
