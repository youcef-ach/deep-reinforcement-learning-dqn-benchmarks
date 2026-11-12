# Archived experiment results

90 runs; three training seeds per configuration; 20 greedy evaluation episodes per run. The table reports the mean and sample standard deviation across the three seed means. These are descriptive statistics, not confidence intervals.

| Environment | Variant | Mean return ± sample SD | Success rate |
| --- | --- | ---: | ---: |
| ALE/Pong-v5 | DQN_hard | -20.93 ± 0.12 | — |
| ALE/Pong-v5 | DQN_no_replay | -21.00 ± 0.00 | — |
| ALE/Pong-v5 | DQN_no_target | -21.00 ± 0.00 | — |
| ALE/Pong-v5 | DQN_soft | -21.00 ± 0.00 | — |
| ALE/Pong-v5 | Double_DQN | -21.00 ± 0.00 | — |
| ALE/Pong-v5 | Dueling_DQN | -21.00 ± 0.00 | — |
| ALE/Pong-v5 | Dueling_Double_DQN | -20.98 ± 0.03 | — |
| CartPole-v1 | DQN_hard | 400.78 ± 171.85 | — |
| CartPole-v1 | DQN_no_replay | 358.27 ± 143.64 | — |
| CartPole-v1 | DQN_no_target | 500.00 ± 0.00 | — |
| CartPole-v1 | DQN_soft | 429.97 ± 121.30 | — |
| CartPole-v1 | Double_DQN | 327.87 ± 149.09 | — |
| CartPole-v1 | Dueling_DQN | 384.67 ± 199.76 | — |
| CartPole-v1 | Dueling_Double_DQN | 329.00 ± 149.21 | — |
| LunarLander-v3 | DQN_hard | -29.57 ± 20.82 | — |
| LunarLander-v3 | DQN_no_replay | -67.94 ± 122.18 | — |
| LunarLander-v3 | DQN_no_target | 42.26 ± 178.96 | — |
| LunarLander-v3 | DQN_soft | 119.15 ± 130.59 | — |
| LunarLander-v3 | Double_DQN | -38.83 ± 23.35 | — |
| LunarLander-v3 | Dueling_DQN | -39.84 ± 39.77 | — |
| LunarLander-v3 | Dueling_Double_DQN | 49.84 ± 126.60 | — |
| TreasureGrid-v0 | DQN | -1.87 ± 0.43 | 0.0% |
| TreasureGrid-v0 | Double_DQN | -2.01 ± 0.66 | 0.0% |
| TreasureGrid-v0 | Dueling_DQN | -1.87 ± 0.43 | 0.0% |
| TreasureGrid-v1 | DQN | -1.26 ± 0.57 | 8.3% |
| TreasureGrid-v1 | Double_DQN | -0.44 ± 1.05 | 10.0% |
| TreasureGrid-v1 | Dueling_DQN | -0.34 ± 0.54 | 11.7% |
| TreasureGrid-v2 | DQN | -6.85 ± 1.00 | 3.3% |
| TreasureGrid-v2 | Double_DQN | -6.50 ± 0.64 | 6.7% |
| TreasureGrid-v2 | Dueling_DQN | -6.48 ± 0.39 | 5.0% |

## Correction to the historical report

The original 10-page course report is preserved as a historical submission. Its TreasureGrid narrative claims 100% success and near-optimal paths; the saved results contradict that claim. The actual configuration means range from 0% to 11.7% success. Pong remains near −21 return. Use this table and the raw runs for quantitative claims.

Each run stores its configuration, budget, training trace and evaluation returns. Results have not been reproduced from distributed checkpoints; checkpoints are not included. See provenance.json for original and published SHA-256 hashes.
