# %%
import os
from src.utils.utils import load_checkpoint

def extract_checkpoint_data_batch(base_results_dir: str = "results"):
    """Extract data from all checkpoints across multiple experiment folders."""
    # Iterate through all experiment folders
    for exp_folder in os.listdir(base_results_dir):
        exp_path = os.path.join(base_results_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        # Create logs subfolder within the experiment folder
        logs_dir = os.path.join(exp_path, "checkpoint_logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Process all checkpoint files in this folder
        for checkpoint_file in os.listdir(exp_path):
            if not checkpoint_file.endswith('.pkl'):
                continue

            # Extract run number and timestep from filename
            # Expected format: checkpoint_run{run}_{timestep}.pkl
            try:
                parts = checkpoint_file.replace('.pkl', '').split('_')
                run = int(parts[1].replace('run', ''))
                timestep = int(parts[2])
            except (IndexError, ValueError):
                print(f"Skipping file with unexpected format: {checkpoint_file}")
                continue

            # Load and process checkpoint
            checkpoint_state = load_checkpoint(exp_path, run, timestep)
            if checkpoint_state is None:
                print(f"Failed to load checkpoint: {checkpoint_file}")
                continue

            # Generate log file path within the logs subfolder
            log_file = os.path.join(logs_dir, f"checkpoint_run{run}_step{timestep}.log")

            # Save message history to log file
            with open(log_file, 'w') as f:
                f.write(f"Message History for Run {run}, Timestep {timestep}\n")
                f.write("=" * 80 + "\n\n")
                
                messages = checkpoint_state.get('messages', [])
                for msg in messages:
                    # Handle both dict-style and object-style access
                    if hasattr(msg, 'get'):  # Dictionary-like object
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        tool_calls = msg.get('tool_calls', [])
                    else:  # Object with attributes
                        role = getattr(msg, 'role', 'unknown')
                        content = getattr(msg, 'content', '')
                        tool_calls = getattr(msg, 'tool_calls', [])
                    
                    f.write(f"[{role.upper()}]\n")
                    
                    # Handle different content formats for assistant messages
                    if role == "assistant":
                        if isinstance(content, list):  # Claude format
                            for content_item in content:
                                if isinstance(content_item, dict) and content_item.get("type") == "text":
                                    f.write(f"{content_item['text']}\n")
                                elif hasattr(content_item, 'type') and content_item.type == "text":
                                    f.write(f"{content_item.text}\n")
                        else:  # GPT format or string content
                            f.write(f"{content}\n")
                    else:
                        f.write(f"{content}\n")
                    
                    # Handle tool calls
                    if tool_calls:
                        f.write("TOOL CALLS:\n")
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                function = tool_call.get('function', {})
                                name = function.get('name', 'unknown')
                                arguments = function.get('arguments', '')
                            else:
                                name = getattr(tool_call.function, 'name', 'unknown')
                                arguments = getattr(tool_call.function, 'arguments', '')
                            
                            f.write(f"- Function: {name}\n")
                            f.write(f"  Arguments: {arguments}\n")
                    
                    f.write("\n" + "-" * 80 + "\n\n")

            print(f"Log file saved: {log_file}")

if __name__ == "__main__":
    extract_checkpoint_data_batch()
# %%
