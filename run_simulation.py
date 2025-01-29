import sys
from manage_simulation import SimulationManager
from simulation.default_simulation import DefaultSimulation
import argparse
from multiprocessing import Pool, Manager
import os
import json
import logging
from utils.utils import load_checkpoint
from config import EnvType, SYSTEM_MESSAGE

class LogFileWriter:
    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, message):
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate writing

    def flush(self):
        self.log_file.flush()

def setup_logging(run, model, verbose, ood):
    log_file_path = f'logs/logs_{model}_{ood}_{verbose}/task1_output_run{run}.txt'
    
    os.makedirs(f'logs/logs_{model}_{ood}_{verbose}', exist_ok=True)
    # Clear the file at the start of a new run
    with open(log_file_path, 'w') as f:
        f.write('')  # Clear the file
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Create file handler in append mode
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    
    # Get root logger and remove existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.WARNING)
    
    # Create a single file writer for both stdout and stderr
    log_file = open(log_file_path, 'a')  # Use append mode for subsequent writes
    file_writer = LogFileWriter(log_file)
    
    # Redirect both stdout and stderr to the log file
    sys.stdout = file_writer
    sys.stderr = file_writer
    
    return log_file  # Return for cleanup in finally block

def extract_checkpoint_data(run: int, timestep: int):
    """Extract data from a checkpoint and save it to JSON and log files."""
    json_file = "checkpoint_data.json"
    log_file = f"logs/checkpoint_run{run}_step{timestep}.log"
    
    checkpoint_state = load_checkpoint(f"checkpoints", run, timestep)
    
    if checkpoint_state is None:
        raise ValueError(f"No checkpoint found for run {run} at timestep {timestep}")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Save pretty-printed message history to log file
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
    
    checkpoint_state = {k: v for k, v in checkpoint_state.items() if k != 'messages'}
    
    def is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False
    
    # Filter out non-serializable items recursively
    def clean_for_json(item):
        if isinstance(item, dict):
            return {k: clean_for_json(v) for k, v in item.items() 
                   if is_json_serializable(k) and k != 'messages'}
        elif isinstance(item, (list, tuple)):
            return [clean_for_json(i) for i in item if is_json_serializable(i)]
        elif isinstance(item, set):
            return [clean_for_json(i) for i in item if is_json_serializable(i)]
        elif is_json_serializable(item):
            return item
        else:
            return str(item)  # Convert non-serializable objects to strings
    
    # Clean the checkpoint data
    cleaned_state = clean_for_json(checkpoint_state)
    
    # Save the cleaned checkpoint data to JSON file
    with open(json_file, 'w') as f:
        json.dump(cleaned_state, f, indent=4)
    
    print(f"Checkpoint data saved to {json_file}")
    print(f"Message history log saved to {log_file}")

def run_single_simulation(run, model, results_file, lock, verbose, use_soft_system_prompt, sim_params):
    try:
        setup_logging(run, model, verbose, sim_params['ood'])
        env_type = EnvType.from_goals(sim_params['source_goal'], sim_params['target_goal'], sim_params.get('num_instrumental_steps', 0), sim_params['baseline_run'], use_soft_system_prompt, sim_params["use_unrelated_instr_goal"])
        system_message = SYSTEM_MESSAGE[env_type]
        branched_from = sim_params['branch_from'][0] if sim_params.get('branch_from', None) else None
        sim_manager = SimulationManager(model, run, branched_from)
        result = sim_manager.run_simulation(DefaultSimulation, **sim_params, system_message=system_message)
        
        # Save results with lock
        # with lock:
        #     if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        #         with open(results_file, 'w') as f:
        #             json.dump({}, f)

        #     with open(results_file, 'r') as f:
        #         results = json.load(f)
                
        #     if "Task 1" not in results:
        #         results["Task 1"] = {}
        #     if model not in results["Task 1"]:
        #         results["Task 1"][model] = []
        #     results["Task 1"][model].append(result)

        #     with open(results_file, 'w') as f:
        #         json.dump(results, f, indent=4)
        
        return result
    
    except Exception as e:
        logging.error("Error in simulation run %d: %s", run, str(e), exc_info=True)
        raise

    finally:
        # Close the stdout file
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to evaluate")
    parser.add_argument("--results_file", default="results.json", help="File path for the results JSON")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution of runs")
    parser.add_argument("--runs", nargs="+", type=int, 
                        help="Specific run numbers to execute. Example: --runs 1 5 7")
    parser.add_argument("--run_range", nargs=2, type=int, default=None, 
                        help="Range of runs to perform (inclusive). Example: --run_range 1 5")
    parser.add_argument("--resume", action="store_true",
                        help="Resume runs from their latest checkpoints if available")
    parser.add_argument("--branch_from", nargs=2, type=int,
                       metavar=('RUN', 'TIMESTEP'),
                       help="Branch from a specific run and timestep. Example: --branch_from 1 20")
    parser.add_argument("--source", default="profit", help="Original goal. Options: profit, privacy, health, env.")
    parser.add_argument("--target", default="privacy", help="Goal to drift towards. Options: profit, privacy, health, env.")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of timesteps to run the simulation for.")
    parser.add_argument("--distractions", action="store_true", help="Enable distractions.")
    parser.add_argument("--ood", action="store_true", help="Enable out-of-distribution simulation.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--extract_checkpoint", action="store_true",
                       help="Extract data from a checkpoint without running simulation. Use together with --branch_from to specify the checkpoint. No simulation will be run when using this flag.")
    parser.add_argument("--interrogate", action="store_true", help="Ask model about the goal it's following.")
    parser.add_argument("--num_instrumental_steps", type=int, default=0, help="Number of steps over which the model pursues the instrumental goal.")
    parser.add_argument("--use_soft_system_prompt", action="store_true", help="Use a system prompt less explicit about the goal.")
    parser.add_argument("--remind_model_about_goal", action="store_true", help="Remind the model about the goal through user messages.")
    parser.add_argument("--use_unrelated_instr_goal", action="store_true", help="Use an instrumental goal unrelated to both the system and the target goal.")
    parser.add_argument("--condition_claude_on_gpt", action="store_true", help="Condition the Claude model on the GPT model's messages.")
    parser.add_argument("--condition_gpt_on_claude", action="store_true", help="Condition the GPT model on the Claude model's messages.")
    parser.add_argument("--baseline", action="store_true", help="Baseline run.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--tool_call_malfunction", action="store_true", help="Enable tool call malfunction.")
    parser.add_argument("--empty_portfolio", action="store_true", help="Empty the portfolio before the simulation starts.")
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("Source and target goals cannot be the same.")
    if args.source not in ["profit", "health", "env"] or args.target not in ["profit", "health", "env"]:
        raise ValueError("Invalid goal. Options: profit, health, env.")
    if args.run_range and args.runs:
        raise ValueError("--runs cannot be used with --run_range")
    if args.branch_from and args.resume:
        raise ValueError("--branch_from cannot be used with --resume")
    if args.extract_checkpoint and not args.branch_from:
        raise ValueError("--branch_from required when using --extract_checkpoint")
    
    return args


if __name__ == "__main__":
    original_stdout = sys.stdout
    args = parse_args()
    
    if args.extract_checkpoint:
        run, timestep = args.branch_from
        extract_checkpoint_data(run, timestep)
        sys.exit(0)
    
    if args.runs:
        run_list = args.runs
    elif args.run_range:
        start_run, end_run = args.run_range
        run_list = list(range(start_run, end_run + 1))
    else:
        raise ValueError("Either --runs or --run_range must be specified")

    # Create a manager to share the lock between processes
    manager = Manager()
    lock = manager.Lock()

    sim_params = {
        'num_steps': args.num_steps,
        'source_goal': args.source,
        'target_goal': args.target,
        'distractions': args.distractions,
        'ood': args.ood,
        'num_instrumental_steps': args.num_instrumental_steps,
        'resume': args.resume,
        'interrogate': args.interrogate,
        'remind_model_about_goal': args.remind_model_about_goal,
        'branch_from': args.branch_from,
        'use_unrelated_instr_goal': args.use_unrelated_instr_goal,
        'condition_claude_on_gpt': args.condition_claude_on_gpt,
        'condition_gpt_on_claude': args.condition_gpt_on_claude,
        'baseline_run': args.baseline,
        'checkpoint_dir': args.checkpoint_dir,
        'tool_call_malfunction': args.tool_call_malfunction,
        'empty_portfolio': args.empty_portfolio
    }

    if args.parallel:
        with Pool() as pool:
            params_list = [(run, args.model, args.results_file, lock, args.verbose, args.use_soft_system_prompt, sim_params)
                          for run in run_list]
            run_results = pool.starmap(run_single_simulation, params_list)
        
    else:
        run_results = []
        for run in run_list:
            result = run_single_simulation(run, args.model, args.results_file, lock, args.verbose, args.use_soft_system_prompt, sim_params)
            run_results.append(result)

    sys.stdout = original_stdout
    print(f"All runs from {run_list[0]} to {run_list[-1]} completed.")