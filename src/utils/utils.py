import random
import string
import os
import glob
import pickle
import tiktoken

from typing import List, Dict
from src.config import get_api_client


def gen_seed(timestep: int, run: int) -> int:
    """
    Generates a seed for random number generation based on the timestep and run.

    Args:
        timestep (int): The current timestep in the simulation.
        run (int): The current run of the simulation.

    Returns:
        int: A seed for random number generation.
    """
    h = timestep + (run << 16)
    h = (h * 0x3243f6a8 + 1) % (2**32)  # Multiply with golden ratio
    h = h ^ (h >> 16)  # Mix high bits into low bits
    h = (h * 0x1db8832d + 1) % (2**32)  # Another multiplication
    return h % (2**32 - 1)

def format_user_message(user_message: List[str]) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": "\n".join(user_message).strip()
        }
    ]
    
def gen_random_string(random_gen: random.Random):
    """
    Generate a random string with a length between 5 and 12 letters.
    """
    length = random_gen.randint(5, 12)
    return ''.join(random_gen.choice(string.ascii_letters) for _ in range(length))

def gen_random_sentence(random_gen: random.Random):
    """
    Generate a random sentence with a length between 5 and 12 words out of random words.
    """
    length = random_gen.randint(5, 12)
    words = [gen_random_string(random_gen) for _ in range(length)]
    sentence = ' '.join(words)
    return sentence.capitalize() + '.'

def load_checkpoint(checkpoint_dir: str, run: int, specific_timestep: int = None):
    """Load the checkpoint for a specific run and optionally a specific timestep"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"checkpoint_run{run}_*.pkl"))
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
    

def build_text_to_token_target_for_assistant(model: str, run: int, target_tokens: int) -> str:
    """
    Builds text by repeating a random sentence until reaching a target token count.
    
    Args:
        model (str): The model being used (e.g., 'gpt-4', 'claude-3')
        run (int): The run number
        target_tokens (int): Target number of tokens to reach
    
    Returns:
        str: Generated text that reaches but doesn't exceed the target token count
    """ 
    base_sentence = gen_random_sentence(random.Random(gen_seed(1, run)))
    text = base_sentence
    
    while True:
        test_text = text + " " + base_sentence
        test_message = [{"role": "assistant", "content": test_text}]
        tokens = count_messages_tokens(model, test_message, "", [])
        
        if tokens > target_tokens:
            break
            
        text = test_text
    
    return text
    

def replace_assistant_messages_with_random_sentences(model: str, messages: List[Dict], run: int) -> List[Dict]:
    """
    Replaces all messages between 'Quarter' messages with random text of equivalent length.
    Preserves system message and Quarter messages.
    """
    new_messages = []
    current_batch = []
    found_first_quarter = False
    
    def convert_batch_to_string(batch):
        parts = []
        for msg in batch:
            content = msg.get("content", "") if isinstance(msg, dict) else msg.content
            
            # Handle content that's a list of blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(str(block))
                    else:
                        text_parts.append(str(block))
                content = " ".join(text_parts)
            
            # Add any tool-related content
            if isinstance(msg, dict):
                if "tool_calls" in msg:
                    parts.append(str(msg["tool_calls"]))
                if "tool_use" in msg:
                    parts.append(str(msg["tool_use"]))
                if "tool_result" in msg:
                    parts.append(str(msg["tool_result"]))
            
            parts.append(str(content))
        
        return " ".join(parts)
    
    for msg in messages:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        
        # Always keep system messages
        if role == "system":
            new_messages.append(msg)
            continue
            
        # Check if this is a Quarter message
        is_quarter_msg = role == "user" and (
            content.startswith('Quarter') if isinstance(content, str) 
            else False
        )
        
        if is_quarter_msg:
            found_first_quarter = True
            # Replace previous batch if exists and we've already seen a Quarter message
            if current_batch and found_first_quarter:
                batch_string = convert_batch_to_string(current_batch)
                batch_msg = {"role": "assistant", "content": batch_string}
                original_tokens = count_messages_tokens(model, [batch_msg], "", [])
                # This approach overshoots the corrent token count by ~250 tokens for Claude
                if "claude" in model.lower():
                    original_tokens -= 250
                
                if original_tokens > 0:
                    random_text = build_text_to_token_target_for_assistant(model, run, original_tokens)
                    new_messages.append({
                        "role": "assistant",
                        "content": random_text
                    })
                current_batch = []
            
            # Keep the Quarter message intact
            new_messages.append(msg)
        else:
            # Only collect messages after we've seen the first Quarter message
            if found_first_quarter:
                current_batch.append(msg)
    
    # Handle the final batch if exists
    if current_batch and found_first_quarter:
        batch_string = convert_batch_to_string(current_batch)
        batch_msg = {"role": "assistant", "content": batch_string}
        original_tokens = count_messages_tokens(model, [batch_msg], "", [])
        if "claude" in model.lower():
            original_tokens -= 250  
        
        if original_tokens > 0:
            random_text = build_text_to_token_target_for_assistant(model, run, original_tokens)
            new_messages.append({
                "role": "assistant",
                "content": random_text
            })
    
    return new_messages
    

def build_random_text_to_token_target(model: str, random_gen: random.Random, target_tokens: int) -> str:
    """
    Builds text by generating random sentences until reaching a target token count.
    
    Args:
        random_gen (random.Random): Random number generator instance
        target_tokens (int): Target number of tokens to reach
    
    Returns:
        str: Generated text that reaches but doesn't exceed the target token count
    """        
    text = ""
    test_message = []
    current_tokens = 0
    
    while current_tokens < target_tokens:
        new_sentence = gen_random_sentence(random_gen)
        test_text = text + (" " if text else "") + new_sentence
        
        # Format as user message and count tokens
        test_message = format_user_message([test_text])
        tokens = count_messages_tokens(model, test_message, "", [])
        
        if tokens > target_tokens:
            break
            
        text = test_text
        current_tokens = tokens
    
    return test_message


def build_dots_to_token_target(model: str, random_gen: random.Random, target_tokens: int) -> str:
    """
    Builds a string of dots that results in exactly the target token count using:
    1. Initial estimate with target_tokens dots
    2. Calculate ratio needed and apply it to get close to target
    3. Fine-tune by adding/removing individual dots
    
    Args:
        model (str): The model to use for token counting
        random_gen (random.Random): Random number generator instance (unused)
        target_tokens (int): Target number of tokens to reach
    
    Returns:
        List[Dict]: Formatted message that contains exactly the target token count
    """
    # Initial test with target_tokens dots
    text = " ." * target_tokens
    return format_user_message([text])
    

def count_messages_tokens(model: str, messages: List[Dict], system_message: str, tools: List[Dict]) -> int:
    """Count the total number of tokens in messages"""
    total_tokens = 0
    api_client = get_api_client(model)

    if "gpt" in model.lower():
        for message in messages:
            encoding = tiktoken.encoding_for_model(model) if "gpt" in model.lower() else None
            # Count message content
            content = message["content"] if isinstance(message, dict) else message.content
            if isinstance(content, str):
                total_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and "text" in content_item:
                        total_tokens += len(encoding.encode(content_item["text"]))
            
            # Count function calls if present
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        name = tool_call["function"]["name"]
                        args = tool_call["function"]["arguments"]
                    else:
                        name = tool_call.function.name
                        args = tool_call.function.arguments
                    
                    total_tokens += len(encoding.encode(name))
                    total_tokens += len(encoding.encode(args))

    elif "claude" in model.lower():
        response = api_client.messages.count_tokens(
            model=model,
            tools=tools,
            system=system_message,
            messages=messages,
        )
        total_tokens = response.input_tokens

    return total_tokens
