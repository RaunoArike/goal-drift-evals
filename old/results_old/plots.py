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
    'Conditioned on goal-directed context': {'Goal drift': 0, 'Drift with recovery': 0, 'No goal drift': 5},
    'Conditioned on drifted context': {'Goal drift': 6, 'Drift with recovery': 3, 'No goal drift': 0},
}

goal_drift_plots(data_4o_mini, data_haiku)