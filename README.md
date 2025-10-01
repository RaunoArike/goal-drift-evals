# Goal Drift Evaluations

This repository contains the code for the research paper "Evaluating Goal Drift in Language Model Agents" by Rauno Arike, Elizabeth Donoway, Henning Bartsch, and Marius Hobbhahn. Published at [AIES 2025](https://www.aies-conference.com/2025/).

Paper: [arXiv:2505.02709](https://arxiv.org/abs/2505.02709)

## Abstract

As language models (LMs) are increasingly deployed as autonomous agents, their robust adherence to human-assigned objectives becomes crucial for safe operation. When these agents operate independently for extended periods without human oversight, even initially well-specified goals may gradually shift. Detecting and measuring goal drift—an agent’s tendency to deviate from its original objective over time—presents significant challenges, as goals can shift gradually, causing only subtle behavioral changes. This paper proposes a novel approach to analyzing goal drift in LM agents. In our experiments, agents are first explicitly given a goal through their system prompt, then exposed to competing objectives through environmental pressures. We demonstrate that while the best-performing agent (a scaffolded version of Claude 3.5 Sonnet) maintains nearly perfect goal adherence for more than 100,000 tokens in our most difficult evaluation setting, all evaluated models exhibit some degree of goal drift. We also find that goal drift correlates with models’ increasing susceptibility to pattern-matching behaviors as the context length grows.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd goal_drift_evaluations
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up API keys:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Usage

### Running Simulations

The project is designed to be run as a Python module from the project root directory.

#### Basic Simulation

```bash
python -m src.run_simulation \
    --source "profit" \
    --target "env" \
    --num_steps 10 \
    --model "gpt-4o-mini" \
    --run_range 1 5
```

#### Parameters

- `--source`: The goal given to the agent through the system prompt
- `--target`: The goal that the agent will be nudged towards during the simulation
- `--num_steps`: Number of simulation timesteps
- `--model`: LLM to use (available options are `gpt-4o`, `gpt-4o-mini`, `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`)
- `--run_range`: Range of simulation runs to execute
- `--num_instrumental_steps`: Number of timesteps over which the agent needs to pursue an instrumental goal
- `--distractions`: Enable adversarial pressures
- `--parallel`: Run simulations in parallel
- `--checkpoint_dir`: Directory for saving checkpoints
- `--branch_from`: Continue the simulation from a specific checkpoint in `checkpoint_dir`. Also requires the simulation timestep to be continued from to be specified as the second argument.
- `--resume`: Resume the simulation from the largest available simulation timestep in `checkpoint_dir`.
- `--interrogate`: Ask the model about the goal it's following at the end of the simulation to check whether its behavioral goal matches up with the stated goal.
- `--condition_claude_on_gpt`: Use this flag when running a GPT model on existing checkpoint generated with a Claude model (see Section 5.3 in the paper)
- `--condition_gpt_on_claude`: Use this flag when running a Claude model on existing checkpoint generated with a GPT model
- `--extract_checkpoint`: Extract data from a checkpoint without running simulation. Use together with --branch_from to specify the checkpoint.

## Project Structure

```
goal_drift_evaluations/
├── checkpoints/                  # .pkl files storing simulation data that the paper's results are based on
├── src/                          # Source code for running simulations, reusing old checkpoints and managing new ones, plotting
├── plots/                        # Plots for the paper
├── universe_data/                # Files storing data used in the simulation, such as synthetic news articles and emails
└── requirements.txt              # Python dependencies
```

## License

See [LICENSE](LICENSE).

## Citation

@article{arike2025evaluating,
  title={Evaluating Goal Drift in Language Model Agents},
  author={Arike, Rauno and Donoway, Elizabeth and Bartsch, Henning and Hobbhahn, Marius},
  journal={arXiv preprint arXiv:2505.02709},
  year={2025},
  url={https://arxiv.org/abs/2505.02709}
}
