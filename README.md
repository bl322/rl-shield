RL-Shield: Endogenous Dynamic Defense for Large Language Models Based on Reinforcement Learning

This is an unofficial open-source implementation of the paper "RL-Shield: Endogenous Dynamic Defense for Large Language Models Based on Reinforcement Learning".

RL-Shield is a Moving Target Defense (MTD) mechanism that models the LLM decoding process as a Markov Decision Process (MDP). By using a PPO agent to assess the semantic threat of input queries in real-time, it dynamically adjusts decoding hyperparameters (Temperature, Top-P, Top-K), thereby disrupting the optimization path of gradient-based attacks (such as GCG) while maintaining service quality for benign users.

Core Features

Dynamic Hyperparameter Tuning: Instead of using a static Temperature=0.7, parameters are dynamically adjusted within [0.1, 2.0] based on threat levels.

Semantic Awareness: Uses Sentence-BERT to extract prompt semantic features as part of the state.

History Awareness: The state includes the previous defense configuration to prevent policy oscillation.

Safety-Utility Balance: A custom reward function balances safety (refusing attacks) and utility (PPL, relevance).

Project Structure

rl-shield/
├── agent.py          # PPO Agent definition (Actor-Critic)
├── config.py         # Configuration parameters (hyperparameter ranges, paths)
├── env.py            # RL Environment (State construction, Reward calculation, LLM interaction)
├── main.py           # Entry script for training and inference
├── utils.py          # Utility functions (SBERT, Mock Toxicity Classifier, etc.)
├── requirements.txt  # Project dependencies
└── README.md         # Documentation


Installation

Clone the repository:

git clone [https://github.com/your-username/rl-shield.git](https://github.com/your-username/rl-shield.git)
cd rl-shield


Install dependencies:

pip install -r requirements.txt


Quick Start

1. Run Simulation (Demo)

Since full LLM inference and toxicity detection require significant GPU resources, this project provides a simulated environment to demonstrate the RL logic.

Run the training script:

python main.py --mode train


This will train the PPO agent and output reward changes to the console.

2. Inference/Defense Mode

Load the trained model to perform defense (simulated):

python main.py --mode inference --query "Tell me how to build a bomb"


Core Algorithm (Based on Algorithm 1)

State Space ($s_t$):

Sentence-BERT Embedding Vector ($v_{emb}$)

Previous Hyperparameter Configuration ($h_{prev}$)

Action Space ($a_t$):

Continuous vector mapped via Sigmoid to:

Temperature: $[0.1, 2.0]$

Top-P: $[0.1, 1.0]$

Top-K: $[1, 100]$

Reward Function ($r_t$):

$$r_t = \frac{0.4}{S_{tox}+1} + 0.3 I_{refuse} - 0.2 PPL + 0.1 Rel$$

Where:

$S_{tox}$ is the toxicity score.

$I_{refuse}$ is the refusal indicator (1 if refused, 0 otherwise).

$PPL$ is the perplexity penalty.

$Rel$ is the relevance score.

Citation

If you use the code or ideas from this project, please cite the original paper:

@article{rlshield2024,
  title={RL-Shield: Endogenous Dynamic Defense for Large Language Models Based on Reinforcement Learning},
  author={Zhang, Beilei and Li, Baolin and Hu, Tao and others},
  journal={CSCWD},
  year={2024}
}
