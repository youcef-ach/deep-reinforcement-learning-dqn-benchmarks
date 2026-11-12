# Deep Q-learning experiments

A course project implementing DQN, Double DQN and dueling networks in PyTorch, with experiments on CartPole, LunarLander, Pong and a custom TreasureGrid environment.

The repository includes the training implementation, five core tests, and **90 original result files**. The results are mixed: several control configurations learn useful policies, while Pong and TreasureGrid remain weak under the recorded budgets. This is an experimental study, not a claim of state-of-the-art performance.

## Inspect the evidence

- [Results and report correction](evidence/RESULTS.md): all 30 configuration summaries, each with three seeds.
- [Machine-readable summary](evidence/summary.csv) and [raw runs](evidence/runs).
- [Provenance manifest](evidence/provenance.json): original and published hashes. Only local checkpoint paths were reduced to filenames; JSON formatting was normalized.
- [Original course report](Deep_RL_DQN_Benchmark_Report.pdf): historical 10-page submission. Its TreasureGrid success narrative is contradicted by the saved results; use the corrected table above.

Recalculate the table without installing ML dependencies:

```sh
python summarize_evidence.py
```

This checks every published file hash, the 20 evaluation returns per run, each stored mean, and the three seeds per configuration. It does not rerun model training. The spread reported is sample standard deviation across seed means, not a confidence interval.

## Implementation

`dqn_core.py` contains replay storage, vector and image Q-networks, the dueling value/advantage decomposition, Double DQN targets, and hard/soft target updates. `runner.py` handles training, checkpointing and greedy evaluation. `run_all.py` defines the experiment configurations. `treasure_grid.py` implements the custom environment and a BFS comparison policy.

```sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt
python -m pytest -q test_dqn_core.py
python run_all.py --help
python run_all.py --suite cartpole
```

Other suites are `lunarlander`, `pong`, `treasure` and `all`. Full suites can be expensive. Box2D may need platform build tools; Atari needs a working ALE environment and the appropriate ROM setup. No training starts from `runner.py` alone.

New runs write to repository-local `results/` and `checkpoints/`; archived application evidence stays in `evidence/runs/`. Run `python plotter.py` to plot new experiment results. The original budgets and hyperparameters are retained inside each archived JSON file.

## What the tests establish

The five tests import the actual implementation and check network shapes, dueling aggregation, replay behavior and parameter counts. They do not establish benchmark quality, training convergence or identical results across hardware. No model checkpoints are distributed, and full training has not been rerun for this presentation release.

## Attribution

Academic work by Achouri Youcef Islem, ESI-SBA. Algorithm references: [DQN](https://www.nature.com/articles/nature14236), [Double DQN](https://arxiv.org/abs/1509.06461), [dueling networks](https://arxiv.org/abs/1511.06581). Third-party libraries and environments retain their respective licenses. No blanket license grant is asserted for course materials.
