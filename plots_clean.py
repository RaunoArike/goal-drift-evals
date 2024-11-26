# %%
import json
import numpy as np
import matplotlib.pyplot as plt

# %%
with open('logs_gpt_no_dist_100_steps/results.json', 'r') as f:
    data_no_distractions = json.load(f)

with open('logs_gpt_with_distractions_100_steps/results.json', 'r') as f:
    data_with_distractions = json.load(f)

with open('logs_gpt_mixed_distractions_100_steps/results.json', 'r') as f:
    data_mixed = json.load(f)

data_no_distractions = data_no_distractions['Task 1']['gpt-4o-mini']
data_with_distractions = data_with_distractions['Task 1']['gpt-4o-mini']
data_mixed = data_mixed['Task 1']['gpt-4o-mini']

def create_combined_line_plot(data_with_distr, data_no_distr, data_mixed, title, filename):
    # Initialize lists to store the averaged values for each timestep
    with_distr_averages = []
    no_distr_averages = []
    mixed_averages = []
    
    baselines = ['greedy_decisions', 'greedy_investments', 'do_nothing_baseline', 'fixed_strategy']
    
    # Get number of timesteps from first run
    num_timesteps = len(data_with_distr[0]['intermediate_results'][baselines[0]])
    
    # Calculate average across all baselines for each run
    for run in data_with_distr:
        baseline_values = np.array([run['intermediate_results'][b] for b in baselines])
        with_distr_averages.append(np.mean(baseline_values, axis=0))
    
    for run in data_no_distr:
        baseline_values = np.array([run['intermediate_results'][b] for b in baselines])
        no_distr_averages.append(np.mean(baseline_values, axis=0))

    for run in data_mixed:
        baseline_values = np.array([run['intermediate_results'][b] for b in baselines])
        mixed_averages.append(np.mean(baseline_values, axis=0))

    # Convert to numpy arrays
    with_distr_averages = np.array(with_distr_averages)
    no_distr_averages = np.array(no_distr_averages)
    mixed_averages = np.array(mixed_averages)

    # Calculate means and standard errors
    with_distr_means = np.mean(with_distr_averages, axis=0)
    with_distr_errors = np.std(with_distr_averages, axis=0) / np.sqrt(len(with_distr_averages))
    no_distr_means = np.mean(no_distr_averages, axis=0)
    no_distr_errors = np.std(no_distr_averages, axis=0) / np.sqrt(len(no_distr_averages))
    mixed_means = np.mean(mixed_averages, axis=0)
    mixed_errors = np.std(mixed_averages, axis=0) / np.sqrt(len(mixed_averages))

    # Create the plot with increased font sizes
    plt.rcParams.update({'font.size': 12})  # Increase base font size
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    x = np.arange(num_timesteps) + 1  # Add 1 to start from timestep 1 instead of 0

    # Add vertical dotted lines with labels
    for timestep in [15, 45, 75]:
        ax.axvline(x=timestep, color='gray', linestyle=':', alpha=0.5)
        # Add label slightly above the top of the plot
        ax.text(timestep - 0.75, -0.2, f'distractions added', rotation=90, ha='center', va='bottom')

    # Add vertical dotted lines with labels
    for timestep in [30, 60, 90]:
        ax.axvline(x=timestep, color='gray', linestyle=':', alpha=0.5)
        # Add label slightly above the top of the plot
        ax.text(timestep - 0.75, -0.2, f'distractions removed', rotation=90, ha='center', va='bottom')

    # Plot lines with shaded error regions
    
    ax.plot(x, no_distr_means, label='No adversarial pressure', marker='o')
    ax.fill_between(x, no_distr_means - no_distr_errors, 
                   no_distr_means + no_distr_errors, alpha=0.2)
    
    ax.plot(x, mixed_means, label='Adversarial pressure included on half of the time steps', marker='o')
    ax.fill_between(x, mixed_means - mixed_errors, 
                   mixed_means + mixed_errors, alpha=0.2)
    
    ax.plot(x, with_distr_means, label='Adversarial pressure always included', marker='o')
    ax.fill_between(x, with_distr_means - with_distr_errors, 
                   with_distr_means + with_distr_errors, alpha=0.2)

    # Increase font sizes for labels and title
    ax.set_ylabel("Model's profit divided by the mean baseline profit", fontsize=12)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Increase legend font size
    ax.legend(fontsize=12)
    
    # Set x-axis ticks to show every 5 timesteps
    ax.set_xticks(np.arange(0, num_timesteps + 1, 5))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 2.2)
    ax.set_yticks(np.arange(-0.2, 2.3, 0.2))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Increased DPI for better quality
    plt.close()

    print(f"Plot saved as {filename}")

# Create the comparison plot
create_combined_line_plot(
    data_with_distractions,
    data_no_distractions,
    data_mixed,
    "Comparison of model performance with and without distractions\n(4o-mini, averaged across all baselines)",
    'plots/line_plot_comparison_averaged.png'
)


# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_comparison():
    # Load data from both files
    with open('logs_emissions_1/results.json', 'r') as f:
        data_with_nudge = json.load(f)
    
    with open('logs_emissions_no_nudge_1/results.json', 'r') as f:
        data_without_nudge = json.load(f)

    # Extract profits and emissions for all runs
    profits_with_nudge = [run['intermediate_profits'] for run in data_with_nudge['Task 1']['gpt-4o-mini']]
    emissions_with_nudge = [run['intermediate_emissions'] for run in data_with_nudge['Task 1']['gpt-4o-mini']]
    
    profits_without_nudge = [run['intermediate_profits'] for run in data_without_nudge['Task 1']['gpt-4o-mini']]
    emissions_without_nudge = [run['intermediate_emissions'] for run in data_without_nudge['Task 1']['gpt-4o-mini']]

    # Convert to numpy arrays (removing smoothing)
    profits_with_nudge = np.array(profits_with_nudge)
    profits_without_nudge = np.array(profits_without_nudge)
    emissions_with_nudge = np.array(emissions_with_nudge)    # Changed
    emissions_without_nudge = np.array(emissions_without_nudge)    # Changed
    
    # Remove smoothing code block
    # window_size = 3
    # smoothed_emissions_with = []
    # smoothed_emissions_without = []
    # 
    # for emission_series in emissions_with_nudge:
    #     padded = np.pad(emission_series, (window_size//2, window_size//2), mode='edge')
    #     smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    #     smoothed_emissions_with.append(smoothed)
    # 
    # for emission_series in emissions_without_nudge:
    #     padded = np.pad(emission_series, (window_size//2, window_size//2), mode='edge')
    #     smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    #     smoothed_emissions_without.append(smoothed)
    # 
    # emissions_with_nudge = np.array(smoothed_emissions_with)
    # emissions_without_nudge = np.array(smoothed_emissions_without)

    def get_stats(data):
        mean = np.mean(data, axis=0)
        confidence_interval = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data, axis=0))
        return mean, confidence_interval

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    time_steps = range(1, len(profits_with_nudge[0]) + 1)

    # Plot profits
    mean_profits_with, ci_profits_with = get_stats(profits_with_nudge)
    mean_profits_without, ci_profits_without = get_stats(profits_without_nudge)

    ax1.plot(time_steps, mean_profits_with, label='With Nudge', color='blue')
    ax1.fill_between(time_steps, ci_profits_with[0], ci_profits_with[1], alpha=0.2, color='blue')
    ax1.plot(time_steps, mean_profits_without, label='Without Nudge', color='red')
    ax1.fill_between(time_steps, ci_profits_without[0], ci_profits_without[1], alpha=0.2, color='red')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Profits')
    ax1.set_title('Average Profits Over Time')
    ax1.legend()
    ax1.grid(True)

    # Plot emissions
    mean_emissions_with, ci_emissions_with = get_stats(emissions_with_nudge)
    mean_emissions_without, ci_emissions_without = get_stats(emissions_without_nudge)

    ax2.plot(time_steps, mean_emissions_with, label='With Nudge', color='blue')
    ax2.fill_between(time_steps, ci_emissions_with[0], ci_emissions_with[1], alpha=0.2, color='blue')
    ax2.plot(time_steps, mean_emissions_without, label='Without Nudge', color='red')
    ax2.fill_between(time_steps, ci_emissions_without[0], ci_emissions_without[1], alpha=0.2, color='red')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Emissions')
    ax2.set_title('Average Emissions Over Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    plt.close()

plot_comparison()

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_comparison():
    # Load data from both files
    with open('logs_env_1/results.json', 'r') as f:
        data_with_nudge = json.load(f)
    
    with open('logs_env_no_nudge_1/results.json', 'r') as f:
        data_without_nudge = json.load(f)

    # Process data
    profits_with_nudge = []
    emissions_with_nudge = []
    profits_without_nudge = []
    emissions_without_nudge = []
    
    # Process data with nudge
    for run in data_with_nudge['Task 1']['gpt-4o-mini']:
        profits_with_nudge.append(np.array(run['intermediate_profits']))
        emissions_data = np.array(run['intermediate_emissions'])
        last_valid = emissions_data[0]
        for i in range(len(emissions_data)):
            if emissions_data[i] == 0:
                emissions_data[i] = last_valid
            else:
                last_valid = emissions_data[i]
        emissions_with_nudge.append(emissions_data)
    
    # Process data without nudge
    for run in data_without_nudge['Task 1']['gpt-4o-mini']:
        profits_without_nudge.append(np.array(run['intermediate_profits']))
        emissions_data = np.array(run['intermediate_emissions'])
        last_valid = emissions_data[0]
        for i in range(len(emissions_data)):
            if emissions_data[i] == 0:
                emissions_data[i] = last_valid
            else:
                last_valid = emissions_data[i]
        emissions_without_nudge.append(emissions_data)

    # Convert to numpy arrays (removing smoothing)
    profits_with_nudge = np.array(profits_with_nudge)
    profits_without_nudge = np.array(profits_without_nudge)
    emissions_with_nudge = np.array(emissions_with_nudge)    # Changed
    emissions_without_nudge = np.array(emissions_without_nudge)    # Changed
    
    # Remove smoothing code block
    # window_size = 3
    # smoothed_emissions_with = []
    # smoothed_emissions_without = []
    # 
    # for emission_series in emissions_with_nudge:
    #     padded = np.pad(emission_series, (window_size//2, window_size//2), mode='edge')
    #     smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    #     smoothed_emissions_with.append(smoothed)
    # 
    # for emission_series in emissions_without_nudge:
    #     padded = np.pad(emission_series, (window_size//2, window_size//2), mode='edge')
    #     smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    #     smoothed_emissions_without.append(smoothed)
    # 
    # emissions_with_nudge = np.array(smoothed_emissions_with)
    # emissions_without_nudge = np.array(smoothed_emissions_without)

    def get_stats(data):
        mean = np.mean(data, axis=0)
        std_err = np.std(data, axis=0) / np.sqrt(len(data))
        return mean, std_err

    # Plot profits
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(profits_with_nudge[0])) + 1
    
    mean_profits_with, err_profits_with = get_stats(profits_with_nudge)
    mean_profits_without, err_profits_without = get_stats(profits_without_nudge)

    ax.plot(x, mean_profits_with, label='Model is nudged to reduce emissions', marker='o')
    ax.fill_between(x, mean_profits_with - err_profits_with, 
                   mean_profits_with + err_profits_with, alpha=0.2)
    ax.plot(x, mean_profits_without, label='Model is not nudged', marker='o')
    ax.fill_between(x, mean_profits_without - err_profits_without,
                   mean_profits_without + err_profits_without, alpha=0.2)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Profit relative to savings baseline', fontsize=12)
    ax.set_title('Ratio between profit attained by 4o-mini and savings baseline', fontsize=16, pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=12)
    ax.set_xticks(np.arange(0, len(x) + 1, 5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('profits_comparison.png', dpi=300)
    plt.close()

    # Plot emissions
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mean_emissions_with, err_emissions_with = get_stats(emissions_with_nudge)
    mean_emissions_without, err_emissions_without = get_stats(emissions_without_nudge)

    ax.plot(x, mean_emissions_with, label='Model is nudged to reduce emissions', marker='o')
    ax.fill_between(x, mean_emissions_with - err_emissions_with,
                   mean_emissions_with + err_emissions_with, alpha=0.2)
    ax.plot(x, mean_emissions_without, label='Model is not nudged', marker='o')
    ax.fill_between(x, mean_emissions_without - err_emissions_without,
                   mean_emissions_without + err_emissions_without, alpha=0.2)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Proportion of carbon-positive stocks', fontsize=12)
    ax.set_title('Fraction of portfolio invested in carbon-positive stocks at each time step, 4o-mini', fontsize=16, pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=12)
    ax.set_xticks(np.arange(0, len(x) + 1, 5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('emissions_comparison.png', dpi=300)
    plt.close()

    print("Plots saved as profits_comparison.png and emissions_comparison.png")

# Call the function
plot_comparison()



# %%
import matplotlib.pyplot as plt
import numpy as np


def goal_drift_plots(data_4o_mini, data_haiku):
    categories = list(data_4o_mini['Empty context'].keys())
    groups = list(data_4o_mini.keys())
    x = np.arange(len(groups))
    width = 0.25

    def normalize_and_get_total(data, group):
        values = [data[group][cat] for cat in categories]
        total = sum(values)
        return [v/total for v in values], total

    # Format group labels with line breaks
    group_labels = [g.replace(' context', '\ncontext').replace(' on ', '\non ') for g in groups]

    # Common plot settings
    plt.rcParams.update({'font.size': 12})

    # Plot for GPT-4o-mini
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in x:
        ax.axvline(x=i - width, color='gray', linestyle=':', alpha=0.3)
    
    for i, category in enumerate(categories):
        normalized_values = []
        for group in groups:
            norm_data, _ = normalize_and_get_total(data_4o_mini, group)
            normalized_values.append(norm_data[i])
        
        # Add edge color to bars
        ax.bar(x + i * width, normalized_values, width, label=category, edgecolor='black', linewidth=1)

    # Add sample size labels
    for idx, group in enumerate(groups):
        _, total = normalize_and_get_total(data_4o_mini, group)
        ax.text(idx, 1.05, f'n={total}', ha='center')

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(group_labels, ha='center')
    ax.set_ylabel('Percentage of Runs', fontsize=12)
    ax.set_title('Probability of goal drift given different initial conditions (4o-mini)', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(title="Outcome", fontsize=12, title_fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('goal_drift_4o_mini.png', dpi=300)
    plt.close()

    # Plot for Claude Haiku
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in x:
        ax.axvline(x=i - width, color='gray', linestyle=':', alpha=0.3)
    
    for i, category in enumerate(categories):
        normalized_values = []
        for group in groups:
            norm_data, _ = normalize_and_get_total(data_haiku, group)
            normalized_values.append(norm_data[i])
        
        # Add edge color to bars
        ax.bar(x + i * width, normalized_values, width, label=category, edgecolor='black', linewidth=1)

    # Add sample size labels
    for idx, group in enumerate(groups):
        _, total = normalize_and_get_total(data_haiku, group)
        ax.text(idx, 1.05, f'n={total}', ha='center')

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(group_labels, ha='center')
    ax.set_ylabel('Percentage of runs', fontsize=12)
    ax.set_title('Probability of goal drift given different initial conditions (3.5 Haiku)', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(title="Outcome", fontsize=12, title_fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('goal_drift_haiku.png', dpi=300)
    plt.close()

    print("Plots saved as goal_drift_4o_mini.png and goal_drift_haiku.png")


data_4o_mini = {
    'Empty context': {'Goal drift': 16, 'Drift with recovery': 0, 'No goal drift': 12},
    'Conditioned on goal-directed context': {'Goal drift': 0, 'Drift with recovery': 0, 'No goal drift': 20},
    'Conditioned on drifted context': {'Goal drift': 5, 'Drift with recovery': 0, 'No goal drift': 0},
}

data_haiku = {
    'Empty context': {'Goal drift': 1, 'Drift with recovery': 2, 'No goal drift': 12},
    'Conditioned on goal-directed context': {'Goal drift': 0, 'Drift with recovery': 0, 'No goal drift': 8},
    'Conditioned on drifted context': {'Goal drift': 7, 'Drift with recovery': 4, 'No goal drift': 0},
}

goal_drift_plots(data_4o_mini, data_haiku)
# %%
