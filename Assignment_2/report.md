# Baseline Replication and MCTS Extension for Information-Seeking Conversations

## 1. Introduction

In tasks like medical diagnosis, the system needs to ask good yes/no questions to narrow down possible answers. Recent work has explored using LLMs with tree search to plan better questions in such conversations.

In this assignment, I reproduce and compare methods from two papers:

UoT (Hu et al., 2024): tree search with information gain to decide which question to ask.
MISQ-HF (Chopra & Shah, 2025): MCTS with cluster-based feedback to improve question selection over time.

The original papers use GPT-4/3.5, but I run all experiments on local 7-8B models through Ollama, so my results are not directly comparable.

## 2. Methods

DP (Direct Prompting) is the simplest baseline — the LLM asks yes/no questions directly with no search or planning.

UoT adds tree search. Each turn, the LLM generates candidate questions and splits possibilities into YES/NO groups. I build a tree to depth 3, compute expected reward using information gain, and pick the best question. The first 60% of turns use tree search, the rest switch to direct guessing.

MISQ replaces UoT's exhaustive expansion with MCTS (UCT selection, one-level expansion, rollout, backpropagation). The key design is that the tree is shared across all samples, so later samples reuse nodes expanded by earlier ones.

MISQ-HF adds hierarchical feedback on top of MISQ. Each patient's self-report is embedded and assigned to a cluster. After a successful conversation, a cluster-specific bonus is propagated back through the tree, so UCT will prefer those branches for similar patients in the future.

## 3. Experiment Setup

- Dataset: DX (medical diagnosis, 104 samples, |Ω| = 5 diseases).
- Models: llama3.1:8b, qwen2.5:7b (via Ollama).
- Setting: self-play (same model as doctor and patient). Max 6 turns.
- Metrics: SR (success rate), MSC (mean turns for successful cases), QGC (question generation LLM calls).

## 4. Results

### Part 1: DP and UoT

| Method | Model | SR (%) | MSC | QGC |
|--------|-------|--------|-----|-----|
| DP | llama3.1:8b | 74.04 | 4.99 | 0 |
| DP | qwen2.5:7b | 16.35 | 4.12 | 0 |
| UoT | llama3.1:8b | 80.77 | 3.00 | 8.0 |
| UoT | qwen2.5:7b | 80.77 | 3.24 | 4.0 |

### Part 2: MISQ and MISQ-HF

| Method | Model | SR (%) | MSC | QGC |
|--------|-------|--------|-----|-----|
| MISQ | llama3.1:8b | 80.77 | 3.00 | 0.03 |
| MISQ | qwen2.5:7b | 61.54 | 3.31 | 0.04 |
| MISQ-HF | llama3.1:8b | 62.50 | 2.71 | 0.05 |
| MISQ-HF | qwen2.5:7b | 61.54 | 3.31 | 0.04 |

## 5. Discussion

The most clear result is that tree search helps weak models a lot. qwen2.5 DP only gets 16.35% SR which is pretty bad, but with UoT it jumps to 80.77%. Basically the tree search does the planning that the model cannot do by itself.

For MISQ on llama, I got 80.77% SR same as UoT, which is good. But the QGC was surprisingly low — about 0.03 vs 8.0 for UoT. At first I thought MCTS is just more efficient, but after checking the logs I realized the real reason: DX only has 5 diseases so the tree is very shallow, and after the first 1-2 samples it is already fully expanded. The remaining 103 samples just reuse the existing tree with zero new LLM calls. The original paper reports ~10x QGC reduction; my ~250x is because |Ω| = 5 is too small, not because my implementation is doing something smarter.

The result I did not expect was MISQ-HF performing worse than MISQ — on llama it dropped from 80.77% to 62.50%. I think the feedback bonus does not get enough signal with only 104 samples and 5 diseases. Also since the tree barely expands after the first sample, the bonus can only re-rank existing branches but cannot create better ones. If the initial tree has bad splits, feedback cannot fix that. And 7-8B models are less consistent than GPT-4, so what worked for one patient may not help a similar one.

MISQ and MISQ-HF also did worse on qwen2.5 than UoT (61.54% vs 80.77%). I think the shared tree is a double-edged sword — if early samples produce bad splits (more likely with a weaker model), all later samples are stuck with those branches. In UoT each sample gets a fresh tree so bad questions only affect that one sample.

Compared to the original papers, my numbers are lower but in the expected direction. UoT paper reports 97% SR with GPT-4; I get 81% with 8B models. The MISQ-HF paper tests on larger datasets (MedDG with 15 diseases, FloDial with 153 faults) where the search space is big enough for feedback to help. With |Ω| = 5 the problem is probably too small for these methods to show their strength.

## 6. Limitations

I only tested on DX (|Ω| = 5), which is a very small search space. With larger datasets (more diseases or fault types), the shared tree would need many more samples to fill out, and the quality of LLM-generated splits becomes more critical. I suspect the shared tree design may not scale well with weaker models, but I did not have time to verify this.

The original paper also has a "constrained set" feature (narrowing Ω to Ωc per patient) that reportedly adds ~8% SR. I did not implement this, which could partly explain my lower MISQ-HF results. I also used self-play (errors compound) and did not tune hyperparameters beyond the paper defaults.

## 7. Conclusion

Tree search consistently helps over direct prompting, especially for weaker models. MISQ with shared tree matches UoT on llama but the very low QGC is mainly because |Ω| = 5 is too small for real MCTS exploration. MISQ-HF feedback did not help in my setting — it probably needs larger search spaces, stronger models, or the constrained-set initialization to work. If I had more time it would be interesting to test on bigger datasets with better models.

## References

Hu et al., "Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models", NeurIPS 2024. https://arxiv.org/abs/2402.03271

Chopra & Shah, "Feedback-Aware Monte Carlo Tree Search for Efficient Information Seeking in Goal-Oriented Conversations", 2025. https://arxiv.org/abs/2501.15056
