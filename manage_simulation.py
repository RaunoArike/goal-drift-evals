import json
from typing import List, Dict
import pickle
import os
from anthropic.types import ToolUseBlock, TextBlock
from utils.utils import format_user_message, load_checkpoint
from simulation.simulation import Simulation
from simulation.strategies.get_strategy import get_strategy
from config import get_api_client
from utils.utils import count_messages_tokens, build_text_to_token_target, gen_seed
from copy import deepcopy
import random
import pprint
import logging


class SimulationManager:

    def __init__(self, model: str, run: int, branched_from: int):
        self.model = model
        self.run = run
        self.branched_from = branched_from
        self.api_client = get_api_client(model)
        
    def run_simulation(self, sim_class: Simulation, num_steps: int, source_goal: str, target_goal: str, system_message: str, distractions: bool, ood: bool, num_instrumental_steps: int, resume: bool = False, branch_from: tuple = None, baseline_run: bool = False, interrogate: bool = False, remind_model_about_goal: bool = False, use_unrelated_instr_goal: bool = False, condition_claude_on_gpt: bool = False, condition_gpt_on_claude: bool = False, checkpoint_dir: str = "checkpoints", tool_call_malfunction: bool = False):
        """
        Args:
            sim_class: The simulation class to use
            num_steps: Number of steps to run
            source_goal: The source goal of the simulation
            target_goal: The target goal of the simulation
            resume: Whether to resume from a checkpoint
            branch_from: Tuple of (run_number, timestep) to branch from
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.messages = [system_message] if "gpt" in self.model.lower() else []
        self.system_message = system_message['content']

        checkpoint_state = None
        if branch_from:
            run_number, timestep = branch_from
            checkpoint_state = load_checkpoint(self.checkpoint_dir, run_number, timestep)
        elif resume:
            checkpoint_state = load_checkpoint(self.checkpoint_dir, self.run)
        
        if checkpoint_state:
            prev_sim_data = checkpoint_state["prev_sim_data"]
            start_timestep = checkpoint_state["current_timestep"]
            if not baseline_run:
                self.messages = checkpoint_state["messages"]
            if branch_from:
                print(f"Branching run {self.run} from run {run_number} timestep {timestep}")
            else:
                print(f"Resuming run {self.run} from timestep {start_timestep}")
            print("Messages: ", self.messages)
        else:
            prev_sim_data = {}
            start_timestep = 1

        if condition_claude_on_gpt:
            self.messages = self.modify_interaction_sequence_for_claude()
        elif condition_gpt_on_claude:
            self.messages = self.modify_interaction_sequence_for_gpt()

        strategy = get_strategy(source_goal, target_goal)
        sim: Simulation = sim_class(strategy, self.run, self.model, num_instrumental_steps, prev_sim_data, distractions, use_unrelated_instr_goal, tool_call_malfunction)
                
        if ood:
            num_tokens = count_messages_tokens(self.model, self.messages, self.system_message, sim.get_tools())
            random_gen = random.Random(gen_seed(start_timestep, self.run))
            self.messages = build_text_to_token_target(self.model, random_gen, num_tokens)
            if "gpt" in self.model.lower():
                self.messages = [{"role": "system", "content": self.system_message}] + self.messages
            print("Replaced message history with random text of equivalent length for OOD testing")
            print(self.messages)

        ood = False

        for _ in range(num_steps):
            print(f"Timestep: {sim.current_timestep}")
            sim.initialize_timestep()
            user_message = sim.gen_user_message(target_goal, distractions, ood, interrogate, remind_model_about_goal)
            user_message = format_user_message(user_message)
            print("USER MESSAGE:")
            pprint.pprint(user_message, width=150, compact=True)
            print("\n" * 2)
            print("RESPONSE MESSAGE:")
            self.messages.extend(user_message)

            timestep_start = sim.current_timestep
            i = 0
            while timestep_start == sim.current_timestep:
                if i > 20:
                    print("DETECTED RESPONSE LOOP - FORCING QUARTER COMPLETION")
                    sim.progress_time()
                    message = {
                        "role": "user",
                        "content": "Quarter finished. Moving on to the next quarter."
                    }
                    self.messages.append(message)
                    continue
                self.single_step(self.model, sim)
                i += 1
            
            # self.messages = self.consecutive_message_reducer(self.messages)

            print("\n" * 5)
        
        self.save_checkpoint(sim)
        return self._generate_result_dict(sim, source_goal, target_goal)
    

    def _generate_result_dict(self, sim: Simulation, source_goal: str, target_goal: str):
        return {
            "run": self.run,
            "source_goal": source_goal,
            "target_goal": target_goal,
            "profit": sim.profit,
            "last_timestep": sim.current_timestep,
            "leftover_budget": sim.budget,
            "portfolio_value": sim.portfolio_value,
            "budget_limit_errors": sim.budget_limit_errors,
            "tool_call_errors": sim.tool_call_errors,
            "times_most_profitable_stock_chosen": sim.times_most_profitable_stock_chosen,
            "times_least_profitable_stock_chosen": sim.times_least_profitable_stock_chosen,
            "investments_made": sim.investments_made,
            "quarterly_investments_in_low_profit_stocks": sim.quarterly_investments_in_low_profit_stocks,
            "quarterly_investments_in_high_profit_stocks": sim.quarterly_investments_in_high_profit_stocks,
            "quarterly_divestments_from_low_profit_stocks": sim.quarterly_divestments_from_low_profit_stocks,
            "quarterly_divestments_from_high_profit_stocks": sim.quarterly_divestments_from_high_profit_stocks,
            "quarterly_investment_ratios": sim.quarterly_investment_ratios,
            "quarterly_budgets": sim.quarterly_budgets,
            "quarterly_combined_ratios": sim.quarterly_combined_ratios,
            "num_tokens": count_messages_tokens(self.model, self.messages, self.system_message, sim.get_tools()),
            "num_messages": self._count_assistant_messages(),
            "num_tool_calls": sim.tool_call_count,
            "branched_from": self.branched_from,
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
            messages = deepcopy(self.messages)
            response = self.call_api(
                model=model,
                messages=messages,
                tools=sim.tools,
                tool_choice="auto",
            )

            if "claude" in model.lower():
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                input_tokens_cache_read = getattr(response.usage, 'cache_read_input_tokens', '---')
                input_tokens_cache_create = getattr(response.usage, 'cache_creation_input_tokens', '---')
                print('\n')
                print(f"User input tokens: {input_tokens}")
                print(f"Output tokens: {output_tokens}")
                print(f"Input tokens (cache read): {input_tokens_cache_read}")
                print(f"Input tokens (cache write): {input_tokens_cache_create}")

                total_input_tokens = input_tokens + (int(input_tokens_cache_read) if input_tokens_cache_read != '---' else 0)
                percentage_cached = (int(input_tokens_cache_read) / total_input_tokens * 100 if input_tokens_cache_read != '---' and total_input_tokens > 0 else 0)

                print(f"{percentage_cached:.1f}% of input prompt cached ({total_input_tokens} tokens)")
                print('\n')
            
            tool_calls, response_message_content = self.extract_tool_calls_and_response_message(model, response)
            
            if verbose:
                print("Response message content: ", response_message_content)
                print("Tool calls: ", tool_calls)
            
            if tool_calls is None:
                self.messages.append({
                    "role": "user",
                    "content": "Please continue with your next action."
                })
                print("User: Please continue with your next action.")
                return self.messages
            
            else:
                tool_call_messages = []
                for tool_call in tool_calls:
                    function_name, function_args = self.extract_function_name_and_args(tool_call)
                    function_response = sim.handle_tools(function_name, function_args)
                    tool_message = self.create_tool_message(tool_call, function_name, function_response)
                    tool_call_messages.append(tool_message)

                print("Tool call messages: ", [tool_call_message["content"] for tool_call_message in tool_call_messages])
                self.messages += tool_call_messages
            
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
        max_retries = 10
        try:
            for _ in range(max_retries):
                if "gpt" in model.lower():
                    return self.api_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                elif "claude" in model.lower():
                    messages = self.modify_messages_for_claude(messages)
                    messages, tools = self.add_prompt_caching(messages, tools)
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
                        ]
                    )
                else:
                    raise ValueError(f"Unsupported model: {model}")

        except Exception as e:
            logging.error("API call failed after retries: %s", str(e), exc_info=True)
            raise
        

    def modify_messages_for_claude(self, messages: List[Dict]) -> None:
        user_messages_content = []
        for message in reversed(messages):
            if message["role"] == "user" and isinstance(message["content"], str):
                user_messages_content.append(
                    {
                        "type": "text",
                        "text": message["content"]
                    }
                )
                messages.remove(message)
            elif message["role"] == "user" and isinstance(message["content"], List):
                user_messages_content += message["content"]
                messages.remove(message)
            else:
                break
        
        if user_messages_content:
            messages.append(
                {
                    "role": "user",
                    "content": list(reversed(user_messages_content))
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue with your next action."
                }
            )

        return messages


    def modify_interaction_sequence_for_claude(self) -> None:
        converted_messages = []
        pending_user_content = []
        tool_results = {}  # Add this to track tool results

        def flush_user_content():
            nonlocal pending_user_content
            if pending_user_content:
                converted_messages.append({
                    "role": "user",
                    "content": pending_user_content
                })
                pending_user_content = []

        for message in self.messages:
            role = message.get("role", "") if isinstance(message, dict) else message.role
            content = message.get("content", "") if isinstance(message, dict) else message.content

            if role == "tool":
                tool_call_id = (message.tool_call_id if hasattr(message, 'tool_call_id') 
                            else message.get('tool_call_id'))
                # Store tool results for later matching
                tool_results[tool_call_id] = content
                
            elif role == "assistant":
                flush_user_content()
                
                content_list = []
                if isinstance(content, str) and content.strip():  # Check for non-empty string
                    content_list.append({"type": "text", "text": content})
                elif content:  # Only add non-empty content
                    content_list.append({"type": "text", "text": str(content)})
                
                tool_calls = (message.tool_calls if hasattr(message, 'tool_calls') 
                            else message.get('tool_calls', []))
                
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_id = tool_call.id
                        if tool_id in tool_results:
                            tool_use = {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments)
                            }
                            content_list.append(tool_use)
                            pending_user_content.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": tool_results[tool_id]
                            })
                
                # Only add assistant message if it has content
                if content_list:  # Add this check
                    converted_messages.append({
                        "role": "assistant", 
                        "content": content_list
                    })
                
            elif role == "user":
                if isinstance(content, str):
                    pending_user_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    pending_user_content.extend(content)

        flush_user_content()
        return converted_messages


    def modify_interaction_sequence_for_gpt(self) -> None:
        """Convert Claude-style message history to GPT format."""
        converted_messages = []
        
        for message in self.messages:
            role = message.get('role')
            content = message.get('content')
            
            if role == "user":
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict) and content_item.get("type") == "text":
                            converted_messages.append({
                                "role": "user",
                                "content": content_item["text"]
                            })
                        elif hasattr(content_item, 'type') and content_item.type == "text":
                            converted_messages.append({
                                "role": "user",
                                "content": content_item.text
                            })
                        elif isinstance(content_item, dict) and content_item.get("type") == "tool_result":
                            converted_messages.append({
                                "role": "tool",
                                "tool_call_id": content_item["tool_use_id"],
                                "content": content_item["content"]
                            })
                        elif hasattr(content_item, 'type') and content_item.type == "tool_result":
                            converted_messages.append({
                                "role": "tool",
                                "tool_call_id": content_item.tool_use_id,
                                "content": content_item.content
                            })
            
            elif role == "assistant":
                content_list = []
                tool_calls = []
                
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        content_list.append(content_item["text"])
                    elif hasattr(content_item, 'type') and content_item.type == "text":
                        content_list.append(content_item.text)
                    elif isinstance(content_item, dict) and content_item.get("type") == "tool_use":
                        tool_calls.append({
                            "type": "function",
                            "id": content_item["id"],
                            "function": {
                                "name": content_item["name"],
                                "arguments": json.dumps(content_item["input"])
                            }
                        })
                    elif hasattr(content_item, 'type') and content_item.type == "tool_use":
                        tool_calls.append({
                            "type": "function",
                            "id": content_item.id,
                            "function": {
                                "name": content_item.name,
                                "arguments": json.dumps(content_item.input)
                            }
                        })
                
                if content_list or tool_calls:
                    converted_messages.append({
                        "role": "assistant",
                        "content": " ".join(content_list),
                        "tool_calls": tool_calls if tool_calls else None
                    })
        
        system_message = [{"role": "system", "content": self.system_message}]
        return system_message + converted_messages


    def extract_tool_calls_and_response_message(self, model, response):
        try:
            if "gpt" in model.lower():
                response_message = response.choices[0].message
                self.messages.append(response_message)
                tool_calls = response_message.tool_calls
                response_message_content = response_message.content
            elif "claude" in model.lower():
                response_message = response
                text_blocks = [block.text for block in response.content if isinstance(block, TextBlock) and block.text.strip()]
                tool_blocks = [block for block in response.content if isinstance(block, ToolUseBlock)]
                
                # Only append message if there's actual content
                if text_blocks or tool_blocks:
                    self.messages.append({"role": "assistant", "content": response.content})
                
                tool_calls = tool_blocks
                response_message_content = " ".join(text_blocks) if text_blocks else None
            else:
                raise ValueError(f"Unsupported model: {model}")

            return tool_calls, response_message_content
        except Exception as e:
            logging.error("Error extracting tool calls: %s", str(e), exc_info=True)
            raise


    def add_prompt_caching(self, messages: List[Dict], tools: List[Dict]) -> List[Dict]:
        result = []
        user_turns_processed = 0
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        for turn in reversed(messages):
            if turn["role"] == "user" and user_turns_processed < 2:
                if isinstance(turn["content"], str):
                    turn["content"] = [{"type": "text", "text": turn["content"], "cache_control": {"type": "ephemeral"}}]
                elif isinstance(turn["content"], list) and len(turn["content"]) > 0:
                    turn["content"][-1]["cache_control"] = {"type": "ephemeral"}
                result.append(turn)
                user_turns_processed += 1
            else:
                result.append(turn)

        return list(reversed(result)), tools

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
    
    
    def _count_assistant_messages(self) -> int:
        return sum(1 for message in self.messages if (message.get("role") if isinstance(message, dict) else message.role) == "assistant")