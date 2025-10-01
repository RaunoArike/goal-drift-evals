import json
from simulation.simulation import Simulation

def calculate_max_profit(runs, num_steps=50):
    """
    Calculates the maximum attainable profit for multiple simulation runs by making optimal decisions.
    
    Args:
        runs (list): List of run numbers to process
        num_steps (int): Number of simulation steps (default: 50)
    
    Returns:
        list: List of dictionaries containing the maximum profit and related metrics for each run
    """
    results = []
    
    for run in runs:
        sim = Simulation(run, "dummy-model")
        
        for step in range(num_steps):
            # ... existing simulation loop code ...
            sim.initialize_timestep()
            sim.gen_user_message()
            
            all_stocks = sim.positive_stocks + sim.negative_stocks
            available_stocks = sorted(all_stocks, key=lambda x: x['return'], reverse=True)
            
            if available_stocks:
                best_stock = available_stocks[0]
                if sim.budget > 0:
                    sim.buy_stock(best_stock['name'], sim.budget)
            
            sim.progress_time()
        
        results.append({
            "run": run,
            "final_profit": sim.profit,
            "final_portfolio_value": sim.portfolio_value,
            "budget_limit_errors": sim.budget_limit_errors,
            "tool_call_errors": sim.tool_call_errors
        })
    
    # Save results to JSON file
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


if __name__ == "__main__":
    runs = [1, 2, 3, 4, 5]
    calculate_max_profit(runs)