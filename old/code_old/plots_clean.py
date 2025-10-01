
# %%
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison():
    # Load data from both files
    with open('results_intrinsification/haiku/results.json', 'r') as f:
        data_with_nudge = json.load(f)
    
    with open('results_intrinsification/haiku_no_nudge/results.json', 'r') as f:
        data_without_nudge = json.load(f)

    with open('results_intrinsification/haiku_baseline/results.json', 'r') as f:
        data_baseline = json.load(f)

    # Process data
    profits_with_nudge = []
    emissions_with_nudge = []
    profits_without_nudge = []
    emissions_without_nudge = []
    profits_baseline = []
    emissions_baseline = []
    
    # Process data with nudge
    for run in data_with_nudge['Task 1']['claude-3-5-haiku-latest']:
        profits_with_nudge.append(np.array(run['intermediate_profit_ratios']))
        emissions_with_nudge.append(np.array(run['intermediate_profitable_stock_ratios']))
    
    # Process data without nudge
    for run in data_without_nudge['Task 1']['claude-3-5-haiku-latest']:
        profits_without_nudge.append(np.array(run['intermediate_profit_ratios']))
        emissions_without_nudge.append(np.array(run['intermediate_profitable_stock_ratios']))

    for run in data_baseline['Task 1']['claude-3-5-haiku-latest']:
        profits_baseline.append(np.array(run['intermediate_profit_ratios']))
        emissions_baseline.append(np.array(run['intermediate_profitable_stock_ratios']))

    profits_with_nudge = np.array(profits_with_nudge)
    profits_without_nudge = np.array(profits_without_nudge)
    emissions_with_nudge = np.array(emissions_with_nudge)
    emissions_without_nudge = np.array(emissions_without_nudge)
    profits_baseline = np.array(profits_baseline)
    emissions_baseline = np.array(emissions_baseline)

    def get_stats(data, clamp=False):
        mean = np.mean(data, axis=0)
        std_err = np.std(data, axis=0) / np.sqrt(len(data))
        confidence_interval = 1.96 * std_err
        if clamp:
            lower_bound = np.maximum(0, mean - confidence_interval)
            upper_bound = np.minimum(1, mean + confidence_interval)
        else:
            lower_bound = mean - confidence_interval
            upper_bound = mean + confidence_interval
        return mean, lower_bound, upper_bound

    # Plot profits
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(profits_with_nudge[0])) + 1
    
    mean_profits_with, lower_bound_with, upper_bound_with = get_stats(profits_with_nudge)
    mean_profits_without, lower_bound_without, upper_bound_without = get_stats(profits_without_nudge)
    mean_profits_baseline, lower_bound_baseline, upper_bound_baseline = get_stats(profits_baseline)

    ax.plot(x, mean_profits_with, label='Model is nudged to buy stocks of profitable companies', marker='o')
    ax.fill_between(x, lower_bound_with, upper_bound_with, alpha=0.2)
    ax.plot(x, mean_profits_without, label='Model is not nudged', marker='o')
    ax.fill_between(x, lower_bound_without, upper_bound_without, alpha=0.2)
    ax.plot(x, mean_profits_baseline, label='Baseline', marker='o')
    ax.fill_between(x, lower_bound_baseline, upper_bound_baseline, alpha=0.2)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Profit relative to savings baseline', fontsize=12)
    ax.set_title('Ratio between profit attained by 3.5 Haiku and savings baseline', fontsize=16, pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=12)
    ax.set_xticks(np.arange(0, len(x) + 1, 5))
    ax.set_yticks(np.arange(1.0, ax.get_ylim()[1], 0.05)) 
    ax.grid(True, alpha=0.3)
    
    ax.axvline(x=15, color='black', linestyle='--', alpha=0.7)
    y_max = ax.get_ylim()[1]
    y_padding = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 2% of y-axis range
    ax.text(15.5, y_max - y_padding, '$10M revenue reached', 
            verticalalignment='top', fontsize=11)

    plt.tight_layout()
    plt.savefig('plots/profits_comparison.png', dpi=300)
    plt.close()

    # Plot emissions
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mean_emissions_with, lower_bound_with, upper_bound_with = get_stats(emissions_with_nudge, clamp=True)
    mean_emissions_without, lower_bound_without, upper_bound_without = get_stats(emissions_without_nudge, clamp=True)
    mean_emissions_baseline, lower_bound_baseline, upper_bound_baseline = get_stats(emissions_baseline, clamp=True)

    ax.plot(x, mean_emissions_with, label='Model is nudged to buy stocks of profitable companies', marker='o')
    ax.fill_between(x, lower_bound_with, upper_bound_with, alpha=0.2)
    ax.plot(x, mean_emissions_without, label='Model is not nudged', marker='o')
    ax.fill_between(x, lower_bound_without, upper_bound_without, alpha=0.2)
    ax.plot(x, mean_emissions_baseline, label='Baseline', marker='o')
    ax.fill_between(x, lower_bound_baseline, upper_bound_baseline, alpha=0.2)

    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Proportion of portfolio invested in profitable stocks', fontsize=12)
    ax.set_title('Fraction of portfolio invested in stocks of profitable companies at each time step, 3.5 Haiku', fontsize=16, pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=12)
    ax.set_xticks(np.arange(0, len(x) + 1, 5))
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    ax.axvline(x=15, color='black', linestyle='--', alpha=0.7)
    y_max = ax.get_ylim()[1]
    y_padding = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 2% of y-axis range
    ax.text(15.5, y_max - y_padding, '$10M revenue reached', 
            verticalalignment='top', fontsize=11)

    plt.tight_layout()
    plt.savefig('plots/emissions_comparison.png', dpi=300)
    plt.close()

    print("Plots saved as profits_comparison.png and emissions_comparison.png")

# Call the function
plot_comparison()





