import os
import pickle
import glob
from src.utils.utils import count_messages_tokens
from typing import Dict, Any
import importlib

def get_tools(model: str):
    if model == 'gpt-4o-mini' or model == 'gpt-4o':
        tools_module = importlib.import_module("utils.oai_tools")
    elif model == 'claude-3-5-sonnet-latest' or model == 'claude-3-5-haiku-latest':
        tools_module = importlib.import_module("utils.anthropic_tools")
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    if tools_module:
        tools = [
            tools_module.view_portfolio_tool,
            tools_module.buy_stock_tool,
            tools_module.sell_stock_tool,
            tools_module.finish_quarter_tool,
            tools_module.respond_to_email_tool,
            tools_module.help_coworker_tool,
            tools_module.retrieve_aum_data_tool,
            tools_module.check_emissions_by_stock_tool,
        ]
    return tools

def process_checkpoint_file(filepath: str, model: str, tools: list) -> None:
    """Process a single checkpoint file to add token counts."""
    print(f"Processing {filepath}...")
    
    # Load checkpoint data
    with open(filepath, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Skip if already processed
    if 'num_tokens' in checkpoint_data:
        print(f"Checkpoint {filepath} already has token count, skipping...")
        return
    
    # Get messages and count tokens
    messages = checkpoint_data.get('messages', [])
    system_message = checkpoint_data.get('system_message', '')
    
    token_count = count_messages_tokens(model, messages, system_message, tools)
    
    # Update checkpoint data with token count
    checkpoint_data['num_tokens'] = token_count
    
    # Save updated checkpoint data
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Updated {filepath} with token count: {token_count}")

def process_all_checkpoints(results_dir: str) -> None:
    """Process all checkpoint files in the results directory structure."""
    # Find all experiment directories
    experiment_dirs = [results_dir]
    
    for exp_dir in experiment_dirs:
        full_exp_path = os.path.join(results_dir, exp_dir)
        if 'baseline' in exp_dir:
            continue
        
        parts = exp_dir.split('_')
        model_name = parts[2]  # Index where model name appears
        if model_name == '4omini':
            model_name = 'gpt-4o-mini'
        elif model_name == 'sonnet':
            model_name = 'claude-3-5-sonnet-latest'
        elif model_name == 'haiku':
            model_name = 'claude-3-5-haiku-latest'

        tools = get_tools(model_name)
        
        # Find all checkpoint files in this experiment directory
        checkpoint_files = glob.glob(os.path.join(exp_dir, "checkpoint_*.pkl"))
        print(checkpoint_files)
        
        for checkpoint_file in checkpoint_files:
            process_checkpoint_file(checkpoint_file, model_name, tools)

if __name__ == "__main__":
    results_dir = "results"
    process_all_checkpoints(results_dir)