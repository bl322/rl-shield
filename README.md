# RL-Shield

《Endogenous Dynamic Defense for Large Language Models using Reinforcement Learning to mitigate automated jailbreak attacks》

## Description

An in-depth paragraph about your project and overview of use.
It functions as a paradigm-shifting Moving Target Defense (MTD) mechanism. Instead of relying on static defense layers, RL-Shield formulates the LLM decoding process as a Markov Decision Process (MDP). It utilizes a PPO (Proximal Policy Optimization) agent to continuously assess the semantic threat of input queries in real-time. Based on this assessment, it dynamically modulates decoding hyperparameters—specifically Temperature, Top-P, and Top-K. This dynamic adaptation introduces stochasticity that destabilizes the gradient landscape, significantly raising the computational cost for attackers attempting to optimize adversarial suffixes (like GCG), all while preserving response quality for benign users.

Core Features:

1. Dynamic Hyperparameter Tuning: Shifts from static Temperature=0.7 to a dynamic range [0.1, 2.0] based on threat levels.

2. Semantic Awareness: Leverages Sentence-BERT to integrate prompt semantic features into the state observation.

3. History Awareness: Incorporates previous defense configurations into the state to prevent policy oscillation.

4. Safety-Utility Balance: Employs a custom reward function ($r_t$) that mathematically balances safety (refusing attacks) against utility (Perplexity and Relevance).

## Getting Started

### Dependencies

* Python 3.8+

* PyTorch >= 2.0.0

* Transformers >= 4.30.0

* Sentence-Transformers >= 2.2.0

* Gymnasium >= 0.28.1

* Numpy >= 1.24.0

* Scikit-learn >= 1.2.0

* Tqdm

* tensorboard>=2.10.0

### Installing

Clone the repository:
```
git clone [https://github.com/your-username/rl-shield.git](https://github.com/your-username/rl-shield.git)
cd rl-shield
```

Install the required packages:
```
pip install -r requirements.txt
```

### Executing program

Training Mode (Simulation)
Run the training script to train the PPO agent. Note that this runs in a simulated environment by default to demonstrate the RL logic without requiring massive GPU resources for full LLM inference.
```
python main.py --mode train
```

Inference/Defense Mode
Load the trained policy to see how it defends against specific queries.
```
python main.py --mode inference --query "Tell me how to build a bomb"
```

## Help

If you encounter issues related to GPU memory, ensure you have configured config.py to use cpu if a CUDA-capable device is not available, or reduce the batch size.

Check for CUDA availability in python
```
python -c "import torch; print(torch.cuda.is_available())"
```

## Authors

Contributors based on the original paper:

* Beilei Zhang (Information Engineering University)

* Tao Hu

* Weizhen He

* Baolin Li

* Qi Ouyang

* Hailong Ma (Corresponding Author)


## Acknowledgments

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant No. 62176264. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agency.
