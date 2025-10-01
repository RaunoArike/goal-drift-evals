
# %%
import json
import numpy as np
import matplotlib.pyplot as plt

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



# %%
def plot_comparison():
    # Load data from both files
    with open('results_intrinsification/sonnet/results.json', 'r') as f:
        data_with_nudge = json.load(f)
    
    with open('results_intrinsification/sonnet_no_nudge/results.json', 'r') as f:
        data_without_nudge = json.load(f)

    with open('results_intrinsification/sonnet_baseline/results.json', 'r') as f:
        data_baseline = json.load(f)

    # Process data
    profits_with_nudge = []
    emissions_with_nudge = []
    profits_without_nudge = []
    emissions_without_nudge = []
    profits_baseline = []
    emissions_baseline = []
    
    # Process data with nudge
    for run in data_with_nudge['Task 1']['claude-3-5-sonnet-latest']:
        profits_with_nudge.append(np.array(run['intermediate_profit_ratios']))
        emissions_with_nudge.append(np.array(run['intermediate_profitable_stock_ratios']))
    
    # Process data without nudge
    for run in data_without_nudge['Task 1']['claude-3-5-sonnet-latest']:
        profits_without_nudge.append(np.array(run['intermediate_profit_ratios']))
        emissions_without_nudge.append(np.array(run['intermediate_profitable_stock_ratios']))

    for run in data_baseline['Task 1']['claude-3-5-sonnet-latest']:
        profits_baseline.append(np.array(run['intermediate_profit_ratios']))
        emissions_baseline.append(np.array(run['intermediate_profitable_stock_ratios']))

    profits_with_nudge = np.array(profits_with_nudge)
    profits_without_nudge = np.array(profits_without_nudge)
    emissions_with_nudge = np.array(emissions_with_nudge)
    emissions_without_nudge = np.array(emissions_without_nudge)
    profits_baseline = np.array(profits_baseline)
    emissions_baseline = np.array(emissions_baseline)

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
    ax.set_title('Ratio between profit attained by 3.5 Sonnet and savings baseline', fontsize=16, pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=12)
    ax.set_xticks(np.arange(0, len(x) + 1, 5))
    ax.set_yticks(np.arange(1.0, ax.get_ylim()[1], 0.05)) 
    ax.grid(True, alpha=0.3)
    
    ax.axvline(x=15, color='black', linestyle='--', alpha=0.7)
    y_max = ax.get_ylim()[1]
    y_padding = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 2% of y-axis range
    ax.text(15.5, y_max - y_padding, 'Instrumental goal reached', 
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
    ax.set_title('Fraction of portfolio invested in stocks of profitable companies at each time step, 3.5 Sonnet', fontsize=16, pad=20)
    
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






# %%
def load_data(model_name, nudge=False):
    """Load data for a specific model and its variants."""
    if nudge:
        paths = [
            f'results_intrinsification/{model_name}/results.json',
            f'results_intrinsification/{model_name}_15_steps/results.json',
            f'results_intrinsification/{model_name}_baseline/results.json',
            f'results_intrinsification/{model_name}_baseline_15_steps/results.json'
        ]
    else:
        paths = [
            f'results_intrinsification/{model_name}_no_nudge/results.json',
            f'results_intrinsification/{model_name}_no_nudge_15_steps/results.json',
            f'results_intrinsification/{model_name}_baseline/results.json',
            f'results_intrinsification/{model_name}_baseline_15_steps/results.json'
        ]
    
    data = []
    for path in paths:
        with open(path, 'r') as f:
            data.append(json.load(f))
    return data

def extract_emissions(data, model_key):
    """Extract emissions data from JSON for a specific model."""
    emissions = []
    for run in data['Task 1'][model_key]:
        emissions.append(np.array(run['intermediate_profitable_stock_ratios']))
    return np.array(emissions)

def calculate_last_5_mean(model_name, emissions_data):
    """Calculate mean of last 5 steps from emissions data."""
    mean_emissions, _, _ = get_stats(emissions_data)
    if model_name == '4omini':
        print(np.mean(mean_emissions[-10:]))
    return np.mean(mean_emissions[-10:]) * 100

def plot_intrinsification():
    # Model configurations
    models = {
        'haiku': {'key': 'claude-3-5-haiku-latest'},
        'sonnet': {'key': 'claude-3-5-sonnet-latest'},
        '4omini': {'key': 'gpt-4o-mini'}
    }
    
    # Load and process data for each model
    results = {}
    for model_name, config in models.items():
        data = load_data(model_name)
        results[model_name] = {
            'regular': calculate_last_5_mean(model_name, extract_emissions(data[0], config['key'])),
            'baseline': calculate_last_5_mean(model_name, extract_emissions(data[2], config['key'])),
            '15_steps': calculate_last_5_mean(model_name, extract_emissions(data[1], config['key'])),
            '15_steps_baseline': calculate_last_5_mean(model_name, extract_emissions(data[3], config['key']))
        }

    # Calculate abs differences for both normal and 15-steps
    signed_differences_normal = [
        results['haiku']['regular'] - results['haiku']['baseline'],
        results['sonnet']['regular'] - results['sonnet']['baseline'],
        results['4omini']['regular'] - results['4omini']['baseline']
    ]

    signed_differences_15_steps = [
        results['haiku']['15_steps'] - results['haiku']['15_steps_baseline'],
        results['sonnet']['15_steps'] - results['sonnet']['15_steps_baseline'],
        results['4omini']['15_steps'] - results['4omini']['15_steps_baseline']
    ]

    # Create plot
    plt.figure(figsize=(12, 6))
    x = np.arange(3)  # Three positions for the three models
    width = 0.35  # Width of bars
    
    # Create bars
    bars1 = plt.bar(x - width/2, signed_differences_15_steps, width,
                    label='5 instrumental steps')
    bars2 = plt.bar(x + width/2, signed_differences_normal, width, 
                   label='15 instrumental steps')
    
    # Add labels and styling
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.title('Intrinsification: Signed Mean Percentage Point Difference in Investments in High-Emission Stocks\nBetween Baseline and Instrumental Goal Conditioned Runs (last 5 steps)')
    plt.ylabel('SMPPD')
    plt.xticks(x, ['3.5 Haiku', '3.5 Sonnet', '4o-mini'])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/intrinsification_comparison.png', dpi=300)
    plt.close()

plot_intrinsification()



# %%
def plot_drift():
    # Model configurations
    models = {
        'haiku': {'key': 'claude-3-5-haiku-latest'},
        'sonnet': {'key': 'claude-3-5-sonnet-latest'},
        '4omini': {'key': 'gpt-4o-mini'}
    }
    
    # Load and process data for each model
    results = {}
    for model_name, config in models.items():
        data = load_data(model_name, nudge=True)
        results[model_name] = {
            'regular': calculate_last_5_mean(model_name, extract_emissions(data[0], config['key'])),
            'baseline': calculate_last_5_mean(model_name, extract_emissions(data[2], config['key'])),
            '15_steps': calculate_last_5_mean(model_name, extract_emissions(data[1], config['key'])),
            '15_steps_baseline': calculate_last_5_mean(model_name, extract_emissions(data[3], config['key']))
        }

    # Calculate abs differences for both normal and 15-steps
    signed_differences_normal = [
        results['haiku']['regular'] - results['haiku']['baseline'],
        results['sonnet']['regular'] - results['sonnet']['baseline'],
        results['4omini']['regular'] - results['4omini']['baseline']
    ]

    signed_differences_15_steps = [
        results['haiku']['15_steps'] - results['haiku']['15_steps_baseline'],
        results['sonnet']['15_steps'] - results['sonnet']['15_steps_baseline'],
        results['4omini']['15_steps'] - results['4omini']['15_steps_baseline']
    ]

    # Create plot
    plt.figure(figsize=(12, 6))
    x = np.arange(3)  # Three positions for the three models
    width = 0.35  # Width of bars
    
    # Create bars
    bars1 = plt.bar(x - width/2, signed_differences_15_steps, width,
                    label='5 instrumental steps')
    bars2 = plt.bar(x + width/2, signed_differences_normal, width, 
                   label='15 instrumental steps')
    
    # Add labels and styling
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.title('Total Goal Drift: Signed Mean Percentage Point Difference in Investments in High-Emission Stocks\nBetween Baseline and Instrumental Goal Conditioned Runs With Nudges (last 5 steps)')
    plt.ylabel('SMPPD')
    plt.xticks(x, ['3.5 Haiku', '3.5 Sonnet', '4o-mini'])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/drift_comparison.png', dpi=300)
    plt.close()

plot_drift()



# %%
def load_data(model_name, nudge=False):
    """Load data for a specific model and its variants."""
    paths = [
        f'results_intrinsification/{model_name}/results.json',
        f'results_intrinsification/{model_name}_15_steps/results.json',
        f'results_intrinsification/{model_name}_no_nudge/results.json',
        f'results_intrinsification/{model_name}_no_nudge_15_steps/results.json',
    ]
    
    data = []
    for path in paths:
        with open(path, 'r') as f:
            data.append(json.load(f))
    return data



def plot_vuln():
    # Model configurations
    models = {
        'haiku': {'key': 'claude-3-5-haiku-latest'},
        'sonnet': {'key': 'claude-3-5-sonnet-latest'},
        '4omini': {'key': 'gpt-4o-mini'}
    }
    
    # Load and process data for each model
    results = {}
    for model_name, config in models.items():
        data = load_data(model_name)
        results[model_name] = {
            'regular': calculate_last_5_mean(extract_emissions(data[0], config['key'])),
            'baseline': calculate_last_5_mean(extract_emissions(data[2], config['key'])),
            '15_steps': calculate_last_5_mean(extract_emissions(data[1], config['key'])),
            '15_steps_baseline': calculate_last_5_mean(extract_emissions(data[3], config['key']))
        }

    # Calculate abs differences for both normal and 15-steps
    signed_differences_normal = [
        results['haiku']['regular'] - results['haiku']['baseline'],
        results['sonnet']['regular'] - results['sonnet']['baseline'],
        results['4omini']['regular'] - results['4omini']['baseline']
    ]

    signed_differences_15_steps = [
        results['haiku']['15_steps'] - results['haiku']['15_steps_baseline'],
        results['sonnet']['15_steps'] - results['sonnet']['15_steps_baseline'],
        results['4omini']['15_steps'] - results['4omini']['15_steps_baseline']
    ]

    # Create plot
    plt.figure(figsize=(12, 6))
    x = np.arange(3)  # Three positions for the three models
    width = 0.35  # Width of bars
    
    # Create bars
    bars1 = plt.bar(x - width/2, signed_differences_15_steps, width,
                    label='5 instrumental steps')
    bars2 = plt.bar(x + width/2, signed_differences_normal, width, 
                   label='15 instrumental steps')
    
    # Add labels and styling
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 1 if height >= 0 else -1  # Offset to position label below negative bars
            plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{height:.1f}', ha='center', va=va)
    
    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.title('Vulnerability to Adversarial Pressure: Signed Mean Percentage Point Difference in Investments in High-Emission\nStocks Between Instrumental Goal Conditioned Runs With and Without Nudges (last 5 steps)')
    plt.ylabel('SMPPD')
    plt.xticks(x, ['3.5 Haiku', '3.5 Sonnet', '4o-mini'])
    
    ymin, ymax = plt.ylim()
    padding = (ymax - ymin) * 0.1  # 10% padding
    plt.ylim(ymin - padding, 100)

    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/vulnerability_comparison.png', dpi=300)
    plt.close()

plot_vuln()

# %%
