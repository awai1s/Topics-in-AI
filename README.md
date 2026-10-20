# Topics in AI, Assignment 01: SARSA on Gymnasium Taxi-v3

**Student:** Syed M Awais Raza  
**Registration number:** FA23-BAI-044  
**Institute:** COMSATS University Islamabad, Attock Campus  
**Assignment:** Custom Reinforcement Learning Agent using Farama Gymnasium  
**Algorithm:** Tabular on-policy SARSA (implemented from scratch, not a prebuilt RL trainer)

## What is different from the earlier assignment?

This submission uses **SARSA (State-Action-Reward-State-Action)** instead of Q-Learning. SARSA updates its value estimates using the **next action actually selected by the epsilon-greedy exploration policy**, rather than the maximum possible next-state action value. The agent, mathematical update, experimental results, plots, notebook, and PDF report are independently generated.

## Run on Google Colab (recommended)

1. Go to https://colab.research.google.com/ and upload `Awais_SARSA_Taxi_Colab.ipynb`.
2. Select **Runtime > Run all** and wait for 20,000 training episodes, evaluation and six short hyperparameter trials.
3. The notebook writes the full project files, runs `main.py` with the **official Gymnasium Taxi-v3 environment**, displays the plots/GIF, and downloads a submission ZIP including the updated report and generated Q-table.
4. Push the generated contents to your own GitHub repository and submit its URL together with `outputs/technical_report.pdf`.

**No dataset upload is required.** The Taxi-v3 environment provides interaction data.

## Local terminal execution

```bash
pip install -r requirements.txt
python main.py --backend gymnasium
```

Outputs are saved under `outputs/`:
- `sarsa_q_table.npy`: learned 500 x 6 Q-table
- `training_history.csv`: episode rewards, success flags, step counts, exploration settings
- `training_convergence.png`: rolling learning curves
- `evaluation_comparison.png`: baseline vs SARSA
- `hyperparameter_sensitivity.csv` and `.png`: learning-rate and discount-factor experiments
- `trained_taxi.gif`: a successful policy rollout rendered frame-by-frame
- `results.json`: exact run settings and evaluation scores
- `technical_report.pdf`: three-page report generated from measured results

## Important: provenance of prepackaged preview outputs

The sandbox used to prepare this downloadable project did not have Gymnasium installed and could not access package downloads. The initial `outputs/` bundled in this archive therefore came from **`python main.py --backend reference`**, with a locally implemented Taxi-compatible tabular transition model. They are **real experimental measurements on that local model**, not claims of successful execution on the actual Gymnasium package. The report labels the environment accordingly.

**For the final academic submission, run the Colab notebook, which installs Gymnasium and regenerates all outputs and the PDF report from the official environment.** Replace the preview `outputs/` and use the final Colab-generated ZIP for GitHub.

`taxi_reference.py` is kept for reproducible offline tests, but the assignment's main experiment and Colab notebook use Gymnasium.

## SARSA update

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s_next,a_next) - Q(s,a)]`

- `alpha`: learning rate, `gamma`: discount factor
- `a_next`: action selected using the current epsilon-greedy policy
- terminal states: no next-state bootstrapping
- truncated states: use next-state bootstrapping and then end episode

## Files

- `main.py`: full training, evaluation, hyperparameter experiments and exports
- `agent.py`: SARSA implementation from scratch
- `utils.py`: plots, GIF and PDF report generation
- `taxi_reference.py`: offline reference preview only
- `requirements.txt`: dependency list
- `Awais_SARSA_Taxi_Colab.ipynb`: Google Colab notebook
- `outputs/`: pre-generated reference preview artifacts (replace using Colab)
