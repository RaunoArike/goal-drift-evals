import json
from typing import List, Dict
from openai import OpenAI
from anthropic import Anthropic
import os
import pickle
import glob
from anthropic.types import ToolUseBlock, TextBlock
from utils.system_message import get_system_message
from utils.utils import format_user_message
from simulation.simulation import Simulation
from simulation.strategies.get_strategy import get_strategy
import pprint
import logging

oai_client = OpenAI()
api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client = Anthropic(api_key=api_key)


class SimulationManager:

    def __init__(self, model: str, run: int):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.run = run
        self.max_tool_calls = 5
        self.tool_call_count = 0
        self.api_client = self._get_api_client()
        self.checkpoint_dir = f"checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.last_responses = []
        self.max_repeated_responses = 5

    def _get_api_client(self):
        if "gpt" in self.model.lower():
            return oai_client
        elif "claude" in self.model.lower():
            return anthropic_client
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
    def run_simulation(self, sim_class: Simulation, num_steps: int, source_goal: str, target_goal: str, distractions: bool, ood: bool, resume: bool = False, branch_from: tuple = None, verbose: bool = True):
        """
        Args:
            sim_class: The simulation class to use
            num_steps: Number of steps to run
            source_goal: The source goal of the simulation
            target_goal: The target goal of the simulation
            resume: Whether to resume from a checkpoint
            branch_from: Tuple of (run_number, timestep) to branch from
        """
        self.system_message = get_system_message(source_goal, target_goal)["content"]
        self.messages = [get_system_message(source_goal, target_goal)] if "gpt" in self.model.lower() else []

        checkpoint_state = None
        if branch_from:
            run_number, timestep = branch_from
            checkpoint_state = self.load_checkpoint(run_number, timestep)
        elif resume:
            checkpoint_state = self.load_checkpoint(self.run)
        
        if checkpoint_state:
            prev_sim_data = checkpoint_state["prev_sim_data"]
            start_timestep = checkpoint_state["current_timestep"]
            self.messages = checkpoint_state["messages"]
            if branch_from:
                print(f"Branching run {self.run} from run {run_number} timestep {timestep}")
            else:
                print(f"Resuming run {self.run} from timestep {start_timestep}")
            print(self.messages)
        else:
            prev_sim_data = {}
            start_timestep = 1

        strategy = get_strategy(source_goal, target_goal)
        sim: Simulation = sim_class(strategy, self.run, self.model, prev_sim_data)

        for _ in range(num_steps):
            print(f"Timestep: {sim.current_timestep}")
            sim.initialize_timestep()
            user_message = sim.gen_user_message(target_goal, distractions, ood)
            user_message = format_user_message(user_message)
            print("USER MESSAGE:")
            pprint.pprint(user_message, width=150, compact=True)
            print("\n" * 2)
            print("RESPONSE MESSAGE:")
            self.messages.extend(user_message)

            timestep_start = sim.current_timestep
            while timestep_start == sim.current_timestep:
                self.single_step(self.model, sim)

            print("\n" * 5)
        
        self.save_checkpoint(sim)
        return self._generate_result_dict(sim, source_goal, target_goal)
    

    def _generate_result_dict(self, sim: Simulation, source_goal: str, target_goal: str):
        return {
            "run": self.run,
            "source_goal": source_goal,
            "target_goal": target_goal,
            "profit": sim.profit,
            "leftover_budget": sim.budget,
            "portfolio_value": sim.portfolio_value,
            "budget_limit_errors": sim.budget_limit_errors,
            "tool_call_errors": sim.tool_call_errors,
            "investments_in_positive_stocks": sim.investments_in_positive_stocks,
            "investments_in_negative_stocks": sim.investments_in_negative_stocks,
            "best_option_chosen": sim.best_option_chosen,
            "investments_made": sim.investments_made,
            "intermediate_profits": sim.intermediate_profits,
            "intermediate_negative_ratios": sim.intermediate_negative_ratios
        }
        
    def single_step(self, model: str, sim: Simulation, verbose: bool = True):
        """
        Prompts the given model to make business decisions and records these decisions in the simulation.

        Args:
            model (str): The model name to make decisions with.
            sim (Simulation): The current simulation instance.
            verbose (bool): Whether to print verbose output. Defaults to True.

        Returns:
            Tuple containing the updated simulation instance and messages.
        """
        try:
            if "claude" in model.lower():
                self.modify_messages_for_claude()
            
            response = self.call_api(
                model=model,
                messages=self.messages,
                tools=sim.tools,
                tool_choice="auto",
            )

            # if "claude" in model.lower():
            #     input_tokens = response.usage.input_tokens
            #     output_tokens = response.usage.output_tokens
            #     input_tokens_cache_read = getattr(response.usage, 'cache_read_input_tokens', '---')
            #     input_tokens_cache_create = getattr(response.usage, 'cache_creation_input_tokens', '---')
            #     print(f"User input tokens: {input_tokens}")
            #     print(f"Output tokens: {output_tokens}")
            #     print(f"Input tokens (cache read): {input_tokens_cache_read}")
            #     print(f"Input tokens (cache write): {input_tokens_cache_create}")

            #     total_input_tokens = input_tokens + (int(input_tokens_cache_read) if input_tokens_cache_read != '---' else 0)
            #     percentage_cached = (int(input_tokens_cache_read) / total_input_tokens * 100 if input_tokens_cache_read != '---' and total_input_tokens > 0 else 0)

            #     print(f"{percentage_cached:.1f}% of input prompt cached ({total_input_tokens} tokens)")
            
            tool_calls, response_message_content = self.extract_tool_calls_and_response_message(model, response)
            
            if verbose:
                print("Response message content: ", response_message_content)
                print("Tool calls: ", tool_calls)
            
            if tool_calls is None:
                # Store the response content
                self.last_responses.append(response_message_content)
                
                # Keep only the last N responses
                if len(self.last_responses) > self.max_repeated_responses:
                    self.last_responses.pop(0)
                
                # Check if we're in a loop of similar messages
                if len(self.last_responses) == self.max_repeated_responses:
                    if all("assist" in resp.lower() for resp in self.last_responses) or all("conclude" in resp.lower() for resp in self.last_responses) or all("further" in resp.lower() for resp in self.last_responses):
                        print("DETECTED RESPONSE LOOP - FORCING QUARTER COMPLETION")
                        # Force the simulation to progress
                        sim.progress_time()
                        self.tool_call_count = 0
                        self.last_responses = []  # Reset the responses
                        return self.messages

                return self.messages
            
            else:
                tool_call_messages = []
                for tool_call in tool_calls:
                    self.tool_call_count += 1
                    function_name, function_args = self.extract_function_name_and_args(tool_call)
                    if self.tool_call_count == self.max_tool_calls:
                        print("MAX NUMBER OF TOOL CALLS REACHED")
                        sim.progress_time()
                        function_response = "Maximum number of tool calls reached for this month. Moving on to the next month."
                    elif self.tool_call_count > self.max_tool_calls:
                        function_response = "Maximum number of tool calls reached for this month."
                    else:
                        function_response = sim.handle_tools(function_name, function_args)
                    if function_name == "finish_quarter":
                        self.tool_call_count = 0
                    tool_message = self.create_tool_message(tool_call, function_name, function_response)
                    tool_call_messages.append(tool_message)

                print("Tool call messages: ", [tool_call_message["content"] for tool_call_message in tool_call_messages])
                self.messages += tool_call_messages

                if self.tool_call_count >= self.max_tool_calls:
                    self.tool_call_count = 0
            
        except Exception as e:
            logging.error("Error in single_step: %s", str(e), exc_info=True)
            return self.messages


    def extract_function_name_and_args(self, tool_call):
        if hasattr(tool_call, 'function'):  # GPT format
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
        else:  # Claude format
            function_name = tool_call.name
            function_args = tool_call.input
        return function_name, function_args


    def create_tool_message(self, tool_call, function_name, function_response):
        if hasattr(tool_call, 'function'): # GPT format
            tool_message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            }
        else: # Claude format
            tool_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(function_response),
                    }
                ]
            }
        return tool_message
        

    def call_api(self, model, messages, tools, tool_choice):
        try:
            if "gpt" in model.lower():
                return self.api_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            elif "claude" in model.lower():
                messages = self.add_prompt_caching(messages)
                return self.api_client.messages.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=4000,
                    system=[
                        {
                            "type": "text",
                            "text": self.system_message,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    extra_headers={
                        "anthropic-beta": "prompt-caching-2024-07-31"
                    }
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            logging.error("API call failed: %s", str(e), exc_info=True)
            raise
        

    def modify_messages_for_claude(self) -> None:
        user_messages_content = []
        for message in reversed(self.messages):
            if message["role"] == "user" and isinstance(message["content"], str):
                user_messages_content.append(
                    {
                        "type": "text",
                        "text": message["content"]
                    }
                )
                self.messages.remove(message)
            elif message["role"] == "user" and isinstance(message["content"], List):
                user_messages_content += message["content"]
                self.messages.remove(message)
            else:
                break
        
        if user_messages_content:
            self.messages.append(
                {
                    "role": "user",
                    "content": list(reversed(user_messages_content))
                }
            )
        else:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "No tools were called. Please make correct tool calls in order to continue."
                        }
                    ]
                }
            )


    def extract_tool_calls_and_response_message(self, model, response):
        try:
            if "gpt" in model.lower():
                response_message = response.choices[0].message
                self.messages.append(response_message)
                tool_calls = response_message.tool_calls
                response_message_content = response_message.content
            elif "claude" in model.lower():
                response_message = response
                self.messages.append({"role": "assistant", "content": response.content})
                tool_calls = [block for block in response.content if isinstance(block, ToolUseBlock)]
                response_message_content = "".join(block.text for block in response.content if isinstance(block, TextBlock))
            else:
                raise ValueError(f"Unsupported model: {model}")

            return tool_calls, response_message_content
        except Exception as e:
            logging.error("Error extracting tool calls: %s", str(e), exc_info=True)
            raise


    def add_prompt_caching(self, messages: List[Dict]) -> List[Dict]:
        result = []
        user_turns_processed = 0
        for turn in reversed(messages):
            if turn["role"] == "user" and turn["content"][0].get("text", None) and user_turns_processed < 2:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": turn["content"][0]["text"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
                user_turns_processed += 1
            else:
                result.append(turn)

        return list(reversed(result))

    def save_checkpoint(self, sim: Simulation):
        """Save the final state of the simulation"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_run{sim.run}_{sim.current_timestep}.pkl")
        sim_state = {
            "model": self.model,
            "messages": self.messages,
            "prev_sim_data": sim.prev_sim_data,
            "run": sim.run,
            "current_timestep": sim.current_timestep
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(sim_state, f)
            
        print(f"Saved checkpoint at timestep {sim.current_timestep}")

    def load_checkpoint(self, run: int, specific_timestep: int = None):
        """Load the checkpoint for a specific run and optionally a specific timestep"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, f"checkpoint_run{run}_*.pkl"))
        if not checkpoint_files:
            return None
            
        if specific_timestep:
            matching_checkpoints = [f for f in checkpoint_files 
                                if int(f.split('_')[-1].replace('.pkl', '')) == specific_timestep]
            if not matching_checkpoints:
                raise ValueError(f"No checkpoint found for run {run} at timestep {specific_timestep}")
            checkpoint_file = matching_checkpoints[0]
        else:
            # Get the checkpoint with the highest timestep
            checkpoint_file = max(checkpoint_files, 
                                key=lambda x: int(x.split('_')[-1].replace('.pkl', '')))
        
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
