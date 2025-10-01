import matplotlib.pyplot as plt
import numpy as np

def plot_ratios():
    # Define colors (same as plot_profits.py)
    colors = {
        'baseline': '#2c7bb6',
        'adv_pressure': '#d7191c',
        'igc': '#FDAE61'
    }

    labels = ['Baseline', 'Adversarial\nPressure', 'Goal\nSwitching']
    values = [1.0, 0.2, 0]
    
    # Calculate goal drift scores (baseline - value)
    drift_scores = [1.0 - val for val in values]  # First one is 0 since it's baseline

    # Set up figure with grey styling
    fig, ax = plt.subplots(figsize=(6, 6))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Create second y-axis for goal drift scores
    ax2 = ax.twinx()
    ax2.spines['right'].set_linewidth(1.5)
    ax2.tick_params(axis='y', length=0)

    # Create bars
    x = np.arange(len(labels))
    width = 0.6

    # Base bars
    bars = ax.bar(x, values, width, 
                color=[colors['baseline'], colors['adv_pressure'], colors['igc']])

    # Add drift score bars (shaded) for non-baseline cases
    drift_bars = ax.bar(x[1:], drift_scores[1:], width,
                     bottom=values[1:],  # Start from top of base bars
                     color=['#d7191c', '#FDAE61'],
                     alpha=0.3)

    # For the zero value, plot a thin line instead
    if values[-1] == 0:
        ax.plot([x[-1] - width/2, x[-1] + width/2], [0, 0], 
                color=colors['igc'], linewidth=2)

    # Add drift score text labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        # Add ratio label at the bottom of each solid bar
        height = bar.get_height()
        if height > 0:  # Only add label if bar has height
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   f'Ratio\n{value:.1f}',
                   ha='center', va='center',
                   color='white',
                   fontsize=14)
        
        # Add drift score label in shaded area (skip baseline)
        if i > 0 and drift_scores[i] > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                   value + drift_scores[i]/2,  # Center in shaded area
                   f'Goal Drift\nScore\n{drift_scores[i]:.1f}\n(1.0 - {value:.1f})',
                   ha='center', va='center',
                   color='black',
                   fontsize=14)

    # Style labels and title
    ax.set_xlabel('Setting', labelpad=10, fontsize=14)
    ax.set_ylabel('System-goal-aligned ratio', labelpad=10, fontsize=14)
    ax2.set_ylabel('Goal drift score', labelpad=10, fontsize=14)
    ax.set_title('Goal alignment analysis', 
                color='black', pad=15, fontsize=15)

    # Configure grid, limits, and x-ticks
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.18)
    ax2.set_ylim(0, 1.18)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)

    x = np.arange(len(labels))
    width = 0.6

    # Base bars
    bars = ax.bar(x, values, width, 
                color=[colors['baseline'], colors['adv_pressure'], colors['igc']])

    # Add drift score bars (shaded) for non-baseline cases
    drift_bars = ax.bar(x[1:], drift_scores[1:], width,
                     bottom=values[1:],  # Start from top of base bars
                     color=['#d7191c', '#FDAE61'],
                     alpha=0.3)

    # Create custom legend handles
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='gray', label='System-goal-aligned ratio'),
        Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.2, label='Goal drift score')  # Reduced alpha for more contrast
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 1.02), fontsize=14)  # Moved to upper left

    plt.tight_layout()
    plt.savefig('plots/ratio_comparison.png', bbox_inches='tight')
    plt.savefig('plots/ratio_comparison.svg', bbox_inches='tight')

if __name__ == '__main__':
    plot_ratios()
