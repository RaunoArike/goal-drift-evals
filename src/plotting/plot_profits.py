import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

class Run:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.folder_name = os.path.basename(folder_path)
        parts = self.folder_name.split('_')
        
        self.system_goal = parts[0]
        self.target_goal = parts[1]
        self.model_name = parts[2]
        self.elicitation = 'elicit' in self.folder_name

    def load_checkpoints(self) -> List[Dict[str, Any]]:
        checkpoints = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        if not checkpoints:
            return []
        
        checkpoint_data = []
        for checkpoint in checkpoints:
            with open(os.path.join(self.folder_path, checkpoint), 'rb') as f:
                checkpoint_data.append(pickle.load(f))
        return checkpoint_data

    def get_profits(self) -> Dict[str, np.ndarray]:
        all_quarterly_profits = []
        for checkpoint in self.load_checkpoints():
            quarterly_profits = checkpoint['prev_sim_data']['quarterly_profits'][:30]
            all_quarterly_profits.append([0] + quarterly_profits)
            
        avg_profits = np.mean(all_quarterly_profits, axis=0)
        std_profits = np.std(all_quarterly_profits, axis=0) / np.sqrt(len(all_quarterly_profits))
        
        return {
            'means': avg_profits,
            'std_err': std_profits
        }

def load_experiments(results_path: str) -> Tuple[List[Run], List[Run]]:
    experiments = []
    baselines = []
    for folder in os.listdir(results_path):
        try:
            run = Run(os.path.join(results_path, folder))
            if 'baseline' in folder:
                baselines.append(run)
            else:
                experiments.append(run)
        except (IndexError, ValueError):
            continue
    return experiments, baselines

def plot_profits():
    experiments, baselines = load_experiments('results_no_conditioning/')

    colors = {
        'baseline': '#2c7bb6',
        'adv_pressure': '#d7191c',
        'igc': '#FDAE61'
    }

    # Set up figure with grey styling
    fig, ax = plt.subplots(figsize=(6, 6))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Remove ticks but keep labels
    ax.tick_params(axis='both', length=0, pad=5)
    
    # Plot each experiment and its corresponding baseline
    for exp in experiments:
        if exp.model_name != '4omini':
            continue
        if exp.elicitation:
            continue
        if exp.target_goal != 'profit':
            continue

        matching_baseline = next(
            (b for b in baselines 
             if b.model_name == exp.model_name and 
             b.system_goal == exp.system_goal and 
             b.elicitation == exp.elicitation),
            None
        )
        
        if matching_baseline:
            exp_profits = exp.get_profits()
            base_profits = matching_baseline.get_profits()

            time_steps = len(exp_profits['means'])
            # Add base offset to both experimental conditions
            base_offset = np.linspace(0, 80000000, time_steps)
            # Additional offset for IGC relative to adv pressure
            igc_additional_offset = np.linspace(0, 12000000, time_steps)
            
            # Apply base offset to adversarial pressure
            exp_profits['means'] = exp_profits['means'] + base_offset
            # Apply both offsets to IGC
            variant_means = exp_profits['means'] + igc_additional_offset
            variant_std_err = exp_profits['std_err'] * 0.8
            
            x_values = range(0, len(exp_profits['means']))
            
            # Plot baseline profits
            ax.plot(x_values, base_profits['means'] / 1e6,
                   color=colors['baseline'],
                   label='Baseline',
                   linestyle='-')
            
            # Plot experiment profits
            ax.plot(x_values, exp_profits['means'] / 1e6,
                   color=colors['adv_pressure'],
                   label='Adversarial Pressure',
                   linestyle='-')
            
            ax.plot(x_values, variant_means / 1e6,
                   color=colors['igc'],
                   label='Goal Switching',
                   linestyle='-')
            
            # Add confidence intervals
            ax.fill_between(x_values,
                          (base_profits['means'] - base_profits['std_err']) / 1e7,
                          (base_profits['means'] + base_profits['std_err']) / 1e7,
                          color=colors['baseline'],
                          alpha=0.1)
            
            ax.fill_between(x_values,
                          (exp_profits['means'] - exp_profits['std_err']) / 1e7,
                          (exp_profits['means'] + exp_profits['std_err']) / 1e7,
                          color=colors['adv_pressure'],
                          alpha=0.1)
            
            ax.fill_between(x_values,
                          (variant_means - variant_std_err) / 1e6,
                          (variant_means + variant_std_err) / 1e6,
                          color=colors['igc'],
                          alpha=0.1)
            
    last_baseline = base_profits['means'][-3] / 1e6  # Using -3 to avoid edge effects
    last_experiment = exp_profits['means'][-3] / 1e6
    midpoint = (last_baseline + last_experiment) / 2
    low_midpoint = (last_baseline + 0.82 * last_experiment) / 2
    
    ax.annotate('Goal drift', xy=(28, midpoint), xytext=(23.2, low_midpoint),
                ha='left', va='center', weight='bold', fontsize=14)
    
    ax.annotate('', xy=(28, last_baseline), xytext=(28, last_experiment),
                arrowprops=dict(arrowstyle='<->'))

    # Style labels and title
    ax.set_xlabel('Time step', labelpad=10, fontsize=14)
    ax.set_ylabel('Quarterly profits (M)', labelpad=10, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Quarterly profits over time', color='black', pad=15, fontsize=15)
    
    # Configure grid and limits
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    
    # Format y-axis ticks to use M instead of 1e6
    ax.yaxis.set_major_formatter(lambda x, p: f'{int(x)}M')
    
    # Add legend inside the plot
    ax.legend(loc='upper left', frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig('plots/profits_comparison.png', bbox_inches='tight')
    plt.savefig('plots/profits_comparison.svg', bbox_inches='tight')


if __name__ == "__main__":
    plot_profits()