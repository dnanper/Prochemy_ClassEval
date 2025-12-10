import os
import re
import random
import subprocess
import concurrent.futures
from tqdm import tqdm
import json
import shutil
import sys
from func_timeout import func_set_timeout
import func_timeout
import importlib
import unittest
import torch

from utils import read_json, write_jsonl
from test_pipeline import AutoTest

def GEN_SOLUTION(task_describe, prompts, model_obj, tokenizer_or_name, model_type="openai", batch_size=8, strategy="holistic"):
    results = []
    if isinstance(tokenizer_or_name, str):
        if strategy == "compositional":
            # Compositional generation: generate each method separately
            for prompt in tqdm(prompts, desc="Compositional Generation"):
                try:
                    method_completions = _gen_compositional(
                        task_describe,
                        prompt,
                        model_obj,
                        tokenizer_or_name,
                        model_type
                    )
                    results.append(method_completions)
                except Exception as e:
                    print(f"Error in compositional generation for {prompt.get('class_name', 'unknown')}: {e}")
                    results.append("")
        elif strategy == "function-only":
            # Function-only generation: generate each method from docstring only (no skeleton)
            for prompt in tqdm(prompts, desc="Function-Only Generation"):
                try:
                    method_completions = _gen_function_only(
                        task_describe,
                        prompt,
                        model_obj,
                        tokenizer_or_name,
                        model_type
                    )
                    results.append(method_completions)
                except Exception as e:
                    print(f"Error in function-only generation for {prompt.get('class_name', 'unknown')}: {e}")
                    results.append("")
        else:
            # Holistic generation: generate entire class at once
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for prompt in prompts:
                    future = executor.submit(
                        _gen_api,
                        task_describe,
                        prompt,
                        model_obj,
                        tokenizer_or_name,
                        model_type,
                        strategy
                    )
                    futures.append(future)
                
                # Wait for all futures and preserve order
                for future in tqdm(futures, desc="API Batch Processing"):
                    results.append(future.result())
    else:
        # HuggingFace model generation
        print(f"Using HuggingFace model with strategy: {strategy}")
        
        if strategy == "compositional":
            # Compositional generation for HF
            for prompt in tqdm(prompts, desc="HF Compositional Generation"):
                try:
                    method_completions = _gen_compositional_hf(
                        task_describe,
                        prompt,
                        model_obj,
                        tokenizer_or_name,
                        use_chat_template=True
                    )
                    results.append(method_completions)
                except Exception as e:
                    print(f"Error in HF compositional generation for {prompt.get('class_name', 'unknown')}: {e}")
                    results.append("")
        elif strategy == "function-only":
            # Function-only generation for HF
            for prompt in tqdm(prompts, desc="HF Function-Only Generation"):
                try:
                    method_completions = _gen_function_only_hf(
                        task_describe,
                        prompt,
                        model_obj,
                        tokenizer_or_name,
                        use_chat_template=True
                    )
                    results.append(method_completions)
                except Exception as e:
                    print(f"Error in HF function-only generation for {prompt.get('class_name', 'unknown')}: {e}")
                    results.append("")
        elif strategy == "full-context":
            # Full-context generation for HF: provide full implementation of other methods
            for prompt in tqdm(prompts, desc="HF Full-Context Generation"):
                try:
                    method_completions = _gen_full_context_hf(
                        task_describe,
                        prompt,
                        model_obj,
                        tokenizer_or_name,
                        use_chat_template=True
                    )
                    results.append(method_completions)
                except Exception as e:
                    print(f"Error in HF full-context generation for {prompt.get('class_name', 'unknown')}: {e}")
                    results.append("")
        else:
            # Holistic generation for HF
            model = model_obj
            tokenizer = tokenizer_or_name
            for i in tqdm(range(0, len(prompts), batch_size), desc="HF Batch Processing"):
                batch_prompts = prompts[i:i + batch_size]
                batch_prompt_texts = [prompt['skeleton'] for prompt in batch_prompts]
                batch_results = _gen_hf(
                    task_describe,
                    batch_prompt_texts,
                    model,
                    tokenizer,
                    use_chat_template=True
                )
                results.extend(batch_results)
    return results
# helper function

def _gen_api(task_describe, prompt, model_obj, model_name, model_type = "openai", strategy="holistic"):
    """Generate completion API"""
    attempts = 0
    response_text = ""
    if (strategy == "holistic"): 
        instruction = f"Please complete the class {prompt['class_name']} in the following code.\n{prompt['skeleton']}"
    elif (strategy == "compositional"):
        # For compositional, skeleton already contains "instruction + skeleton" from _gen_single_method
        instruction = prompt['skeleton']
    elif (strategy == "function-only"):
        # For function-only, only provide method description (no class skeleton)
        instruction = prompt['skeleton']
    while attempts < 3:
        try:
            if model_type == "gemini":
                # Gemini API with safety settings to reduce blocking
                full_prompt = f"{task_describe}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                print(full_prompt)
                print("--- +++++++++++++++++++++++++++++++++++++++++ ---")
                # Completely disable safety settings - no safety_settings parameter
                response = model_obj.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens":  2048*12,
                    }
                )
                # Handle safety filter and blocked responses
                try:                    
                    response_text = response.text or ""
                except Exception as text_error:
                    # Log detailed error information
                    if response.candidates:
                        candidate = response.candidates[0]
                        finish_reason = candidate.finish_reason
                        
                        # Finish reason mapping:
                        # 0 = FINISH_REASON_UNSPECIFIED, 1 = STOP (success), 
                        # 2 = SAFETY (blocked), 3 = RECITATION, 4 = OTHER
                        reason_names = {
                            0: "UNSPECIFIED",
                            1: "STOP", 
                            2: "SAFETY",
                            3: "RECITATION",
                            4: "OTHER",
                            5: "BLOCKLIST"
                        }
                        reason_name = reason_names.get(finish_reason, f"UNKNOWN({finish_reason})")
                        
                        print(f"[ERROR] Gemini blocked response - Finish Reason: {reason_name}")
                        print(f"[ERROR] Class: {prompt.get('class_name', 'Unknown')}")
                        
                        # Try to get safety ratings
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            print(f"[SAFETY RATINGS]:")
                            for rating in candidate.safety_ratings:
                                print(f"  - {rating.category}: {rating.probability}")
                        
                        # Check if there's any partial content
                        if candidate.content.parts:
                            parts_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                            if parts_text:
                                print(f"[INFO] Found partial content ({len(parts_text)} chars), using it")
                                response_text = parts_text
                            else:
                                response_text = ""
                        else:
                            response_text = ""
                    else:
                        print(f"[ERROR] No candidates in response for class {prompt.get('class_name', 'Unknown')}")
                        response_text = ""
            else:
                # OpenAI API
                response = model_obj.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": task_describe},
                        {"role": "user", "content": instruction}
                    ],
                    max_tokens=1000,
                    temperature=0.0
                )
                response_text = response.choices[0].message.content or ""
            # Try to extract code from markdown code blocks first
            match = re.search(r'```(?:python)?(.*?)```', response_text, re.DOTALL)
            if match:
                extracted_code = match.group(1).strip()
                if extracted_code:  # Make sure extracted code is not empty
                    return extracted_code
            
            # If no code block or empty, return full response (might be plain code)
            stripped_response = response_text.strip()
            if stripped_response:
                return stripped_response
            
            # If still empty, continue to retry
            if attempts < 2:
                print(f"[Warning] Empty response on attempt {attempts + 1}, retrying...")
                attempts += 1
                continue
            
            return ""
        except Exception as e:
            attempts += 1
            if attempts < 3:
                print(f"[Retry {attempts}/3] API error: {e}")
            else:
                print(f"[Failed after 3 attempts] API error: {e}")
    
    final_response = response_text.strip() if response_text else ""
    if not final_response:
        print(f"[Error] No valid response after all retries for class {prompt.get('class_name', 'Unknown')}")
    return final_response


def _gen_compositional(task_describe, prompt, model_obj, model_name, model_type="openai", use_parallel=True):
    """
    Generate code compositionally: generate each method separately and combine them.
    EXACT implementation from inference_pipeline.py Compositional strategy.
    
    Args:
        use_parallel: If True, use ThreadPoolExecutor to generate methods in parallel
    """
    class_name = prompt['class_name']
    methods_info = prompt.get('methods_info', [])
    imports = prompt.get('import_statement', [])
    class_constructor = prompt.get('class_constructor', '')
    class_description = prompt.get('class_description', '')
    
    # Add description to constructor (same as InferenceUtil.add_desc_to_init)
    class_init = _add_desc_to_init(class_description, class_constructor)
    imports_text = '\n'.join(imports)
    
    # Prepare method generation tasks
    method_tasks = []
    for method_to_generate in methods_info:
        # Build class skeleton for this method (same as inference_pipeline.py)
        class_text = imports_text + '\n' + class_init
        
        # Gather each method's signature to construct class level skeleton
        for method in methods_info:
            if method['method_name'] == method_to_generate['method_name']:
                continue
            # Use InferenceUtil.get_method_signature logic
            method_signature = _get_method_signature_from_description(
                method['method_description'], 
                method['method_name']
            )
            class_text += method_signature + "\n        pass\n\n"
        
        # Construct prompt (same as inference_pipeline.py)
        method_name = method_to_generate['method_name']
        inst = f"please complete {method_name} method in the following class {class_name}\n\n"
        class_text_desc = class_text + "\n\n    " + method_to_generate['method_description']
        
        method_tasks.append({
            'method_name': method_name,
            'instruction': inst,
            'skeleton': class_text_desc,
            'class_name': class_name
        })
    
    # Generate methods in parallel or sequentially
    if use_parallel and len(method_tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(method_tasks))) as executor:
            futures = []
            for task in method_tasks:
                future = executor.submit(
                    _gen_single_method,
                    task_describe,
                    task,
                    model_obj,
                    model_name,
                    model_type
                )
                futures.append(future)
            
            # Wait for all methods to be generated
            method_completions = [future.result() for future in futures]
    else:
        # Sequential generation
        method_completions = []
        for task in method_tasks:
            method_code = _gen_single_method(
                task_describe,
                task,
                model_obj,
                model_name,
                model_type
            )
            method_completions.append(method_code)
    
    # Post-process: combine all methods into final class
    # This follows the exact logic from inference_pipeline.py post_process
    imports_text = '\n'.join(imports)
    class_init = _add_desc_to_init(class_description, class_constructor)
    final_class_code = imports_text + '\n' + class_init
    
    for i in range(len(method_completions)):
        method_name = methods_info[i]['method_name']
        raw_output = method_completions[i]
        # Extract method code using InferenceUtil.extract_method_code logic
        method_code = _extract_method_code_inference_util(raw_output, method_name)
        final_class_code += '\n\n' + method_code
    
    return final_class_code


def _gen_function_only(task_describe, prompt, model_obj, model_name, model_type="openai", use_parallel=True):
    """
    Generate code function-only: generate each method from docstring only (no class skeleton).
    Similar to compositional but only provides method_description without full class context.
    
    Args:
        use_parallel: If True, use ThreadPoolExecutor to generate methods in parallel
    """
    class_name = prompt['class_name']
    methods_info = prompt.get('methods_info', [])
    imports = prompt.get('import_statement', [])
    class_constructor = prompt.get('class_constructor', '')
    class_description = prompt.get('class_description', '')
    
    # Add description to constructor
    class_init = _add_desc_to_init(class_description, class_constructor)
    imports_text = '\n'.join(imports)
    
    # Prepare method generation tasks - only with docstring
    method_tasks = []
    for method_to_generate in methods_info:
        method_name = method_to_generate['method_name']
        method_description = method_to_generate['method_description']
        
        # Simple instruction with only method description (no class skeleton)
        inst = f"Please complete the following method {method_name} for class {class_name}:\n\n{method_description}"
        
        method_tasks.append({
            'method_name': method_name,
            'instruction': inst,
            'skeleton': '',  # No skeleton provided for function-only
            'class_name': class_name
        })
    
    # Generate methods in parallel or sequentially
    if use_parallel and len(method_tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(method_tasks))) as executor:
            futures = []
            for task in method_tasks:
                future = executor.submit(
                    _gen_single_function,
                    task_describe,
                    task,
                    model_obj,
                    model_name,
                    model_type
                )
                futures.append(future)
            
            # Wait for all methods to be generated
            method_completions = [future.result() for future in futures]
    else:
        # Sequential generation
        method_completions = []
        for task in method_tasks:
            method_code = _gen_single_function(
                task_describe,
                task,
                model_obj,
                model_name,
                model_type
            )
            method_completions.append(method_code)
    
    # Post-process: combine all methods into final class
    imports_text = '\n'.join(imports)
    class_init = _add_desc_to_init(class_description, class_constructor)
    final_class_code = imports_text + '\n' + class_init
    
    for i in range(len(method_completions)):
        method_name = methods_info[i]['method_name']
        raw_output = method_completions[i]
        # Extract method code
        method_code = _extract_method_code_inference_util(raw_output, method_name)
        final_class_code += '\n\n' + method_code
    
    return final_class_code


def _gen_single_function(task_describe, method_task, model_obj, model_name, model_type):
    """
    Generate a single function from docstring only (no class skeleton).
    Helper function for function-only strategy.
    """
    instruction = method_task['instruction']
    class_name = method_task['class_name']
    
    # Generate method code using API
    method_code = _gen_api(
        task_describe,
        {'class_name': class_name, 'skeleton': instruction},
        model_obj,
        model_name,
        model_type,
        "function-only"
    )
    
    # Return raw output (will be extracted later in post_process)
    return method_code


def _add_desc_to_init(desc, class_init):
    """
    Add description to class constructor.
    Same as InferenceUtil.add_desc_to_init()
    """
    if not desc:
        return class_init
    class_init_list = class_init.split('\n')
    class_init_list[0] += " \n" + desc
    class_init = '\n'.join(class_init_list)
    return class_init


def _get_method_signature_from_description(code, method_name):
    """
    Extract method signature from description.
    Same as InferenceUtil.get_method_signature()
    """
    method_def_prefix = "def " + method_name + '('
    code_segment = code.split('):')
    for segment in code_segment:
        if method_def_prefix in segment:
            return "    " + segment + "):"
    return ""


def _gen_single_method(task_describe, method_task, model_obj, model_name, model_type):
    """
    Generate a single method. Helper function for parallel execution.
    Follows the exact flow from inference_pipeline.py
    """
    method_name = method_task['method_name']
    instruction = method_task['instruction']
    skeleton = method_task['skeleton']
    class_name = method_task['class_name']
    
    # Generate method code using API
    method_code = _gen_api(
        task_describe,
        {'class_name': class_name, 'skeleton': instruction + skeleton},
        model_obj,
        model_name,
        model_type,
        "compositional"
    )
    
    # Return raw output (will be extracted later in post_process)
    return method_code

def _extract_method_code_inference_util(code, method_name):
    """
    Extract method code from generated completion.
    EXACT implementation of InferenceUtil.extract_method_code()
    """
    # Extract code from response markers
    # output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
    output_split_identifier_list = ['assistant\n']
    for identifier in output_split_identifier_list:
        if identifier in code:
            code = code.split(identifier)[1]
            break
    
    # Extract from code blocks
    pattern_list = [r"```python(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]"]
    for pattern in pattern_list:
        code_part = re.findall(pattern, code, re.S)
        if code_part:
            code = code_part[0]
            break
    
    code_list = code.split('\n')
    
    method_code_list = []
    method_def_prefix = "def " + method_name + '('
    skip_line_list = ["```", '\r']
    
    # Find the line with "def methodname(...)"
    for i, line in enumerate(code_list):
        if method_def_prefix in line:
            method_code_list = code_list[i:]
            break
    
    if len(method_code_list) == 0:
        return ""
    
    # Skip unwanted lines
    for i, line in enumerate(method_code_list):
        if line in skip_line_list:
            method_code_list[i] = ""
    
    # Fix indentation if needed
    if len(method_code_list) > 1 and _get_leading_spaces(method_code_list[1]) - _get_leading_spaces(method_code_list[0]) > 4:
        method_code_list[0] = " " * 4 + method_code_list[0]
    
    # Extract only the method (stop at next method/class)
    first_line_leading_space = _get_leading_spaces(method_code_list[0])
    for i, line in enumerate(method_code_list[1:]):
        if _get_leading_spaces(line) <= first_line_leading_space and len(line) > 0:
            method_code_list = method_code_list[:i + 1]
            break
    
    # Normalize indentation to 4 spaces
    for i, line in enumerate(method_code_list):
        method_code_list[i] = ' ' * (4 - first_line_leading_space) + line
    
    # Add @staticmethod if needed
    if 'self' not in method_code_list[0] and 'cls' not in method_code_list[0]:
        method_code_list.insert(0, ' ' * 4 + "@staticmethod")
    
    # Handle incomplete docstrings
    line_notation_mark = 0
    for line in method_code_list:
        if line == " " * 8 + "\"\"\"" or line == " " * 4 + "\"\"\"":
            line_notation_mark = line_notation_mark + 1
    if line_notation_mark % 2 == 1:
        method_code_list.append(" " * 8 + "\"\"\"")
        method_code_list.append(" " * 8 + "pass")
    
    method_code = '\n'.join(method_code_list)
    method_code = method_code.rstrip() + '\n'
    return method_code


def _get_leading_spaces(string):
    """Get number of leading spaces. Same as InferenceUtil.get_leading_spaces()"""
    return len(string) - len(string.lstrip())

# helper function
def _gen_hf(task_describe, prompts, model, tokenizer, use_chat_template = True):
    """Generate batch with HuggingFace"""
    if use_chat_template:
        all_messages = [
            [
                {"role": "system", "content": task_describe},
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]
    else:
        texts = [
            f"### Instruction:\n{task_describe}\n\n{prompt}\n### Response:\n```python"
            for prompt in prompts
        ]
    # print(texts)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=12288
    ).to(model.device)
    # input_ids = tokenizer()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i, output in enumerate(outputs):
        generated_ids = output[inputs['input_ids'].shape[1]:]
        full_text = tokenizer.decode(output, skip_special_tokens=True)
        print("Full Text: ", full_text)
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(response_text)
        # print("Response Text: ", response_text)
        # match = re.search(r'```(?:python)?(.*?)```', full_text, re.DOTALL)
        # if match:
        #     results.append(match.group(1).strip())
        # else:
        #     results.append(response_text)
    return results


def _gen_compositional_hf(task_describe, prompt, model, tokenizer, use_chat_template=True):
    """
    Generate code compositionally using HuggingFace model: generate each method separately and combine them.
    Similar to _gen_compositional but uses HF model instead of API.
    """
    class_name = prompt['class_name']
    methods_info = prompt.get('methods_info', [])
    imports = prompt.get('import_statement', [])
    class_constructor = prompt.get('class_constructor', '')
    class_description = prompt.get('class_description', '')
    
    # Add description to constructor
    class_init = _add_desc_to_init(class_description, class_constructor)
    imports_text = '\n'.join(imports)
    
    # Prepare method generation prompts
    method_prompts = []
    for method_to_generate in methods_info:
        # Build class skeleton for this method
        class_text = imports_text + '\n' + class_init
        
        # Gather each method's signature to construct class level skeleton
        for method in methods_info:
            if method['method_name'] == method_to_generate['method_name']:
                continue
            method_signature = _get_method_signature_from_description(
                method['method_description'], 
                method['method_name']
            )
            class_text += method_signature + "\n        pass\n\n"
        
        # Construct prompt
        method_name = method_to_generate['method_name']
        inst = f"please complete {method_name} method in the following class {class_name}\n\n"
        class_text_desc = class_text + "\n\n    " + method_to_generate['method_description']
        
        method_prompts.append(inst + class_text_desc)
    
    # Generate all methods using HF batch generation
    method_completions = _gen_hf(task_describe, method_prompts, model, tokenizer, use_chat_template)
    
    # Post-process: combine all methods into final class
    imports_text = '\n'.join(imports)
    class_init = _add_desc_to_init(class_description, class_constructor)
    final_class_code = imports_text + '\n' + class_init
    
    for i in range(len(method_completions)):
        method_name = methods_info[i]['method_name']
        raw_output = method_completions[i]
        # Extract method code
        method_code = _extract_method_code_inference_util(raw_output, method_name)
        final_class_code += '\n\n' + method_code
    
    return final_class_code


def _gen_function_only_hf(task_describe, prompt, model, tokenizer, use_chat_template=True):
    """
    Generate code function-only using HuggingFace model: generate each method from docstring only (no skeleton).
    Returns list of method records for method-level evaluation.
    """
    task_id = prompt.get('task_id')
    class_name = prompt['class_name']
    methods_info = prompt.get('methods_info', [])
    imports = prompt.get('import_statement', [])
    class_constructor = prompt.get('class_constructor', '')
    class_description = prompt.get('class_description', '')
    
    # Add description to constructor
    class_init = _add_desc_to_init(class_description, class_constructor)
    imports_text = '\n'.join(imports)
    
    # Prepare method generation prompts - only with docstring
    method_prompts = []
    for method_to_generate in methods_info:
        method_name = method_to_generate['method_name']
        method_description = method_to_generate['method_description']
        
        # Simple instruction with only method description (no class skeleton)
        inst = f"Please complete the following method {method_name} for class {class_name}:\n\n{method_description}"
        
        method_prompts.append(inst)
    
    # Generate all methods using HF batch generation
    method_completions = _gen_hf(task_describe, method_prompts, model, tokenizer, use_chat_template)
    
    # Post-process: create method-level records
    # Each record contains generated method + ground truth other methods
    method_records = []
    
    for i in range(len(method_completions)):
        current_method_name = methods_info[i]['method_name']
        raw_output = method_completions[i]
        
        # Extract generated method code
        generated_method_code = _extract_method_code_inference_util(raw_output, current_method_name)
        
        # Build class code: imports + constructor + all methods
        class_code = imports_text + '\n' + class_init
        
        for method_info in methods_info:
            if method_info['method_name'] == current_method_name:
                # Use generated method
                class_code += '\n\n' + generated_method_code
            else:
                # Use ground truth method
                if 'solution_code' in method_info:
                    class_code += '\n\n' + method_info['solution_code']
                else:
                    # Fallback: signature + pass
                    method_sig = _get_method_signature_from_description(
                        method_info['method_description'], 
                        method_info['method_name']
                    )
                    class_code += '\n\n' + method_sig + '\n        pass'
        
        # Create method record
        method_records.append({
            'task_id': task_id,
            'class_name': class_name,
            'method_name': current_method_name,
            'prediction': generated_method_code,
            'class_code': class_code
        })
    
    return method_records


def _gen_full_context_hf(task_describe, prompt, model, tokenizer, use_chat_template=True):
    """
    Generate code with full context using HuggingFace model: provide full implementation of other methods,
    only hide the body of the target method to generate.
    Returns list of method records for method-level evaluation.
    """
    task_id = prompt.get('task_id')
    class_name = prompt['class_name']
    methods_info = prompt.get('methods_info', [])
    imports = prompt.get('import_statement', [])
    class_constructor = prompt.get('class_constructor', '')
    class_description = prompt.get('class_description', '')
    
    # Add description to constructor
    class_init = _add_desc_to_init(class_description, class_constructor)
    imports_text = '\n'.join(imports)
    
    # Prepare method generation prompts with full context
    method_prompts = []
    for method_to_generate in methods_info:
        # Build class with full implementation of other methods
        class_text = imports_text + '\n' + class_init
        
        # Add ALL methods with their full implementation
        for method in methods_info:
            if method['method_name'] == method_to_generate['method_name']:
                # Target method: only add signature + description, no body
                continue
            else:
                # Other methods: add full implementation from solution_code
                if 'solution_code' in method:
                    class_text += '\n\n' + method['solution_code']
                else:
                    # Fallback: if no solution_code, use signature + pass
                    method_signature = _get_method_signature_from_description(
                        method['method_description'], 
                        method['method_name']
                    )
                    class_text += '\n\n' + method_signature + "\n        pass"
        class_text += '\n\n' + method_to_generate['method_description']
        
        # Construct prompt
        method_name = method_to_generate['method_name']
        inst = f"Please complete the method {method_name} in the following class {class_name}\n\n"
        
        method_prompts.append(inst + class_text)
    
    # Generate all methods using HF batch generation
    method_completions = _gen_hf(task_describe, method_prompts, model, tokenizer, use_chat_template)
    
    # Post-process: create method-level records
    # Each record contains generated method + ground truth other methods
    method_records = []
    
    for i in range(len(method_completions)):
        current_method_name = methods_info[i]['method_name']
        raw_output = method_completions[i]
        
        # Extract generated method code
        generated_method_code = _extract_method_code_inference_util(raw_output, current_method_name)
        
        # Build class code: imports + constructor + all methods
        class_code = imports_text + '\n' + class_init
        
        for method_info in methods_info:
            if method_info['method_name'] == current_method_name:
                # Use generated method
                class_code += '\n\n' + generated_method_code
            else:
                # Use ground truth method
                if 'solution_code' in method_info:
                    class_code += '\n\n' + method_info['solution_code']
                else:
                    # Fallback: signature + pass
                    method_sig = _get_method_signature_from_description(
                        method_info['method_description'], 
                        method_info['method_name']
                    )
                    class_code += '\n\n' + method_sig + '\n        pass'
        
        # Create method record
        method_records.append({
            'task_id': task_id,
            'class_name': class_name,
            'method_name': current_method_name,
            'prediction': generated_method_code,
            'class_code': class_code
        })
    
    return method_records


# Generate solutions and save to the specified directory
def generate_solutions(test_set_path, mutated_prompt_path, output_directory, model_obj, tokenizer_or_name, model_type="openai", batch_size=8, strategy="holistic"):
    """
    Generate solutions using specified strategy.
    
    Args:
        strategy: "holistic" (generate entire class) or "compositional" (generate each method separately with skeleton) or "function-only" (generate each method from docstring only) or "full-context" (generate each method with full implementation of other methods)
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Load mutated prompts - only support .jsonl format
    if not mutated_prompt_path.endswith('.jsonl'):
        raise ValueError(f"mutated_prompt_path must be a .jsonl file, got: {mutated_prompt_path}")
    
    from utils import read_jsonl
    mutated_prompts = {
        task["prompt_id"]: task["mutated_prompt"] 
        for task in read_jsonl(mutated_prompt_path)
    }
    
    all_tasks = list(read_json(test_set_path))
    is_hf_model = not isinstance(tokenizer_or_name, str)
    # log
    print(f"Processing mode: {'HuggingFace (GPU Batch)' if is_hf_model else 'API (Parallel)'}")
    print(f"Generation strategy: {strategy}")
    print(f"Batch size: {batch_size if is_hf_model else 'N/A (using 10 parallel workers)'}")
    
    for prompt_id, task_describe in mutated_prompts.items():
        print(f"\n{'='*60}")
        print(f"Processing prompt_id: {prompt_id}")
        print(f"{'='*60}")
        
        # Prepare task prompts based on strategy
        if strategy == "compositional":
            # For compositional, need full task info including methods_info
            task_prompts = [{
                'task_id': task["task_id"],
                'class_name': task["class_name"],
                'skeleton': task["skeleton"],
                'methods_info': task.get("methods_info", []),
                'import_statement': task.get("import_statement", []),
                'class_constructor': task.get("class_constructor", ""),
                'class_description': task.get("class_description", "")
            } for task in all_tasks]
        elif strategy == "function-only" or strategy == "full-context":
            # For function-only/full-context, need methods_info with solution_code
            task_prompts = [{
                'task_id': task["task_id"],
                'class_name': task["class_name"],
                'methods_info': task.get("methods_info", []),
                'import_statement': task.get("import_statement", []),
                'class_constructor': task.get("class_constructor", ""),
                'class_description': task.get("class_description", "")
            } for task in all_tasks]
        else:
            # For holistic, just need class_name and skeleton
            task_prompts = [{
                'task_id': task["task_id"],
                'class_name': task["class_name"], 
                'skeleton': task["skeleton"]
            } for task in all_tasks]
        
        completions = GEN_SOLUTION(
            task_describe,
            task_prompts,
            model_obj,
            tokenizer_or_name,
            model_type,
            batch_size=batch_size,
            strategy=strategy,
        )
        
        samples = []
        
        # Handle different output formats based on strategy
        if strategy == "function-only" or strategy == "full-context":
            # Method-level output: list of method records
            for completion in completions:
                if isinstance(completion, list):
                    # Each completion is a list of method records
                    samples.extend(completion)
                else:
                    print(f"Warning: Expected list of method records, got {type(completion)}")
        else:
            # Class-level output: one completion per task
            for task, completion in zip(all_tasks, completions):
                if completion:
                    samples.append({
                        'task_id': task['task_id'],
                        'completion': completion
                    })
                else:
                    print(f"No valid completion for task_id: {task['task_id']}")
                    samples.append({
                        'task_id': task['task_id'],
                        'completion': 'dummy'
                    })
    
        # Create safe model name from model identifier
        if isinstance(tokenizer_or_name, str):
            model_name_safe = tokenizer_or_name.split('/')[-1].replace('.', '_').replace('-', '_')
        else:
            model_name_safe = "model"
        
        # Choose output filename based on strategy
        if strategy == "function-only" or strategy == "full-context":
            output_file = os.path.join(
                output_directory, 
                f"method_level_{model_name_safe}_{prompt_id}.jsonl"
            )
            print(f"Saved {len(samples)} method records to {output_file}")
        else:
            output_file = os.path.join(
                output_directory, 
                f"train_set_{model_name_safe}_{prompt_id}.jsonl"
            )
            print(f"Saved {len(samples)}/{len(all_tasks)} samples to {output_file}")
        
        write_jsonl(output_file, samples)

# Evaluate the generated solutions and select the best prompts
def evaluate_and_select_best_prompts(output_folder, train_set_path, prompts_file_path, best_prompt_output_path, strategy="holistic"):
    """
    Evaluate generated solutions using unittest and select the best prompts based on test results.
    
    Args:
        output_folder: folder containing generated solution files
        train_set_path: path to train_set.json (format similar to ClassEval_data.json)
        prompts_file_path: path to mutated_prompts file
        best_prompt_output_path: path to save the best prompts
        strategy: "holistic"/"compositional" (class-level) or "function-only"/"full-context" (method-level)
    """
    # Initialize AutoTest with train_set data
    train_set_name = os.path.splitext(os.path.basename(train_set_path))[0]
    auto_test = AutoTest(train_set_name)
    
    # Dictionary to store results for each prompt_id
    prompt_results = {}
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation of generated solutions (strategy: {strategy})...")
    print(f"{'='*60}\n")
    
    # Change to output_folder for test execution
    original_dir = os.getcwd()
    os.chdir(output_folder)
    
    # Add output_folder to sys.path for importing
    if output_folder not in sys.path:
        sys.path.insert(0, output_folder)
    
    try:
        # Process each generated solution file
        for filename in os.listdir(output_folder):
            if not filename.endswith(".jsonl"):
                continue
                
            # Extract prompt_id from filename
            try:
                prompt_id = int(re.search(r'_(\d+)\.jsonl$', filename).group(1))
            except (IndexError, ValueError, AttributeError):
                print(f"Warning: Could not extract prompt_id from {filename}, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating Prompt ID: {prompt_id} ({filename})")
            print(f"{'='*60}")
            
            jsonl_file_path = os.path.join(output_folder, filename)
            
            # Branch logic based on strategy
            if strategy == "function-only" or strategy == "full-context":
                # Method-level evaluation (custom_test_pipeline logic)
                all_test_results = _evaluate_method_level(
                    jsonl_file_path, prompt_id, auto_test
                )
                prompt_results[prompt_id] = _calculate_method_metrics(all_test_results, auto_test)
                
                print(f"\nPrompt {prompt_id} Results (Method-Level):")
                print(f"  Total Methods: {prompt_results[prompt_id]['total_methods']}")
                print(f"  Success: {prompt_results[prompt_id]['success_count']} ({prompt_results[prompt_id]['success_rate']:.2%})")
                print(f"  Partial Success: {prompt_results[prompt_id]['partial_success_count']} ({prompt_results[prompt_id]['partial_success_rate']:.2%})")
                print(f"  Fail: {prompt_results[prompt_id]['fail_count']} ({prompt_results[prompt_id]['fail_rate']:.2%})")
                print(f"  Error: {prompt_results[prompt_id]['error_count']} ({prompt_results[prompt_id]['error_rate']:.2%})")
            else:
                # Class-level evaluation (original logic)
                all_test_results = _evaluate_class_level(
                    jsonl_file_path, prompt_id, auto_test
                )
                evaluated_results = _evaluate_prompt_results(all_test_results, auto_test)
                prompt_results[prompt_id] = _calculate_metrics(evaluated_results, auto_test)
                
                print(f"\nPrompt {prompt_id} Results (Class-Level):")
                print(f"  Function Success Rate: {prompt_results[prompt_id]['fun_success']:.2%}")
                print(f"  Class Success Rate: {prompt_results[prompt_id]['class_success']:.2%}")
                print(f"  Function Partial Success Rate: {prompt_results[prompt_id]['fun_partial_success']:.2%}")
                print(f"  Class Partial Success Rate: {prompt_results[prompt_id]['class_partial_success']:.2%}")
            
            # Clean up test files for this prompt
            _cleanup_test_files(output_folder, prompt_id)
    
    finally:
        # Return to original directory and cleanup
        os.chdir(original_dir)
        _cleanup_pycache(output_folder)
        _cleanup_all_test_files(output_folder)
    
    # Select best prompts based on combined metric
    if not prompt_results:
        print("\nNo valid results found!")
        return
    
    # Calculate combined score based on strategy
    if strategy == "function-only" or strategy == "full-context":
        # Method-level: only function success + partial success
        for prompt_id in prompt_results:
            success_rate = prompt_results[prompt_id]['success_rate']
            partial_rate = prompt_results[prompt_id]['partial_success_rate']
            prompt_results[prompt_id]['combined_score'] = 0.7 * success_rate + 0.3 * partial_rate
    else:
        # Class-level: weighted by function and class
        for prompt_id in prompt_results:
            fun_success = prompt_results[prompt_id]['fun_success']
            class_success = prompt_results[prompt_id]['class_success']
            prompt_results[prompt_id]['combined_score'] = 0.4 * fun_success + 0.6 * class_success
    
    # Find best prompt(s) - top 3 by combined score
    sorted_prompts = sorted(prompt_results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
    best_prompt_ids = [pid for pid, _ in sorted_prompts[:3]]  # Top 3 prompts
    max_score = sorted_prompts[0][1]['combined_score'] if sorted_prompts else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for prompt_id in sorted(prompt_results.keys()):
        r = prompt_results[prompt_id]
        print(f"\nPrompt ID: {prompt_id}")
        if strategy == "function-only" or strategy == "full-context":
            print(f"  Total Methods: {r['total_methods']}")
            print(f"  Success Rate: {r['success_rate']:.2%}")
            print(f"  Partial Success Rate: {r['partial_success_rate']:.2%}")
        else:
            print(f"  Function Success: {r['fun_success']:.2%}")
            print(f"  Class Success: {r['class_success']:.2%}")
            print(f"  Function Partial Success: {r['fun_partial_success']:.2%}")
            print(f"  Class Partial Success: {r['class_partial_success']:.2%}")
        print(f"  Combined Score: {r['combined_score']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Best Prompt IDs: {best_prompt_ids}")
    print(f"Max Combined Score: {max_score:.4f}")
    print(f"{'='*60}\n")
    
    # Save best prompts
    if best_prompt_ids:
        best_prompts = []
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prompt_data = json.loads(line)
                    if prompt_data.get('prompt_id') in best_prompt_ids:
                        best_prompts.append(prompt_data)
                except json.JSONDecodeError:
                    continue
        
        with open(best_prompt_output_path, 'w', encoding='utf-8') as f:
            for prompt in best_prompts:
                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
        
        print(f"Best prompts saved to: {best_prompt_output_path}")
    
    # Save detailed results
    results_path = os.path.join(output_folder, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {results_path}")


def _evaluate_method_level(jsonl_file_path, prompt_id, auto_test):
    """
    Evaluate method-level output (function-only/full-context).
    Based on custom_test_pipeline.py logic.
    
    Returns: dict {task_id_methodname: {test_class: {errors, failures, testsRun}}}
    """
    result_dict = {}
    
    # Read method records from JSONL
    task_list = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                task_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Generate python test files
    for task in task_list:
        task_id = task['task_id']
        method_name = task['method_name']
        class_code = task['class_code']
        
        # Build test code
        test_code = "import unittest"
        for method in auto_test.eval_data[task_id]['methods_info']:
            if method['method_name'] == method_name:
                test_code += '\\n\\n' + method['test_code']
                break
        
        # Generate .py file
        test_name = f"{task_id}_{method_name}_prompt_{prompt_id}.py"
        test_code_py = class_code + '\\n' + test_code
        with open(test_name, 'w', encoding='utf-8') as f:
            f.write(test_code_py)
    
    # Run unit tests
    for task in tqdm(task_list, desc=f"Testing prompt {prompt_id} (method-level)"):
        task_id = task['task_id']
        method_name = task['method_name']
        test_module_name = f"{task_id}_{method_name}_prompt_{prompt_id}"
        result_key = f"{task_id}_{method_name}"
        
        # Find test_class for this method
        test_class = None
        for method in auto_test.eval_data[task_id]['methods_info']:
            if method['method_name'] == method_name:
                test_class = method['test_class']
                break
        
        if not test_class:
            continue
        
        try:
            res = auto_test.run_unit_test(test_module_name, test_class, f"prompt_{prompt_id}")
            result_dict[result_key] = {
                test_class: {
                    'errors': len(res.errors),
                    'failures': len(res.failures),
                    'testsRun': res.testsRun
                }
            }
        except func_timeout.exceptions.FunctionTimedOut:
            print(f"⏱️  TIMEOUT (30s) for {test_module_name}.{test_class}")
            result_dict[result_key] = {
                test_class: {'errors': 0, 'failures': 0, 'testsRun': 0}
            }
        except Exception as e:
            print(f"❌ ERROR in test for {test_module_name}.{test_class}: {e}")
            result_dict[result_key] = {
                test_class: {'errors': 0, 'failures': 0, 'testsRun': 0}
            }
    
    return result_dict


def _calculate_method_metrics(test_results, auto_test):
    """
    Calculate metrics for method-level testing.
    Based on custom_test_pipeline.cal_metrics() logic.
    
    Returns: dict with success_rate, partial_success_rate, fail_rate, error_rate, total_methods
    """
    success_count = 0
    partial_success_count = 0
    fail_count = 0
    error_count = 0
    total_count = 0
    
    for task_method_key in test_results:
        for test_class in test_results[task_method_key]:
            result = auto_test.get_test_answer(test_results[task_method_key][test_class])
            total_count += 1
            
            if result == 'success':
                success_count += 1
            elif result == 'partial_success':
                partial_success_count += 1
            elif result == 'fail':
                fail_count += 1
            elif result == 'error':
                error_count += 1
    
    return {
        'success_rate': success_count / total_count if total_count > 0 else 0,
        'partial_success_rate': partial_success_count / total_count if total_count > 0 else 0,
        'fail_rate': fail_count / total_count if total_count > 0 else 0,
        'error_rate': error_count / total_count if total_count > 0 else 0,
        'total_methods': total_count,
        'success_count': success_count,
        'partial_success_count': partial_success_count,
        'fail_count': fail_count,
        'error_count': error_count
    }


def _evaluate_class_level(jsonl_file_path, prompt_id, auto_test):
    """
    Evaluate class-level output (holistic/compositional).
    Based on original test_pipeline.py logic.
    
    Returns: dict {task_id: {test_code_name: {test_class: result_item}}}
    """
    # Read completions from jsonl file
    code_list = {}
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                task_id = item['task_id']
                completion = item['completion']
                
                # Prepare code with imports (same as test_pipeline.py logic)
                if task_id in auto_test.eval_data:
                    full_code = '\\n'.join(auto_test.eval_data[task_id]['import_statement']) + '\\n' + completion
                    code_list[task_id] = [full_code]  # List of 1 code per task
            except json.JSONDecodeError:
                continue
    
    # Generate test files for all tasks
    for task_id in code_list:
        test_code = auto_test.eval_data[task_id]['test']
        task_code_list = code_list[task_id]
        auto_test.gen_py_file(f"{task_id}_prompt_{prompt_id}", task_code_list, test_code)
    
    # Run tests for each task
    all_test_results = {}
    for task_id in tqdm(code_list, desc=f"Testing prompt {prompt_id} (class-level)"):
        try:
            test_result = auto_test.test(
                len(code_list[task_id]), 
                f"{task_id}_prompt_{prompt_id}",
                auto_test.eval_data[task_id]['test_classes'], 
                f"prompt_{prompt_id}"
            )
            all_test_results[task_id] = test_result
        except Exception as e:
            print(f"Error testing {task_id}: {e}")
            all_test_results[task_id] = {}
    
    return all_test_results


def _calculate_metrics(evaluated_results, auto_test):
    """
    Calculate metrics using the EXACT same logic as cal_metrics_pass_at_k() in test_pipeline.py
    Uses auto_test.cal_pass_at_k() for proper pass@k calculation with n=1, k=1.
    
    This replicates cal_metrics_pass_at_k logic but uses the actual cal_pass_at_k function.
    """
    # Parameters for pass@k calculation (n=1, k=1 means we generated 1 sample per task)
    n = 1
    k = 1
    
    sum_num = 0
    success_num = 0
    class_success_num = 0
    class_num = 0
    partial_success_num = 0
    partial_success_class_num = 0
    
    # Loop through tasks (same structure as cal_metrics_pass_at_k)
    for task in evaluated_results:
        class_num += 1
        for test_class in evaluated_results[task]:
            try:
                # Function-level metrics (try block in cal_metrics_pass_at_k)
                if evaluated_results[task][test_class]['success'] != 0:
                    # Use the actual cal_pass_at_k function
                    pass_at_k = auto_test.cal_pass_at_k(
                        n, k, evaluated_results[task][test_class]['success'])
                    success_num += pass_at_k
                if evaluated_results[task][test_class]['success'] + evaluated_results[task][test_class]['partial_success'] != 0:
                    # Use the actual cal_pass_at_k function
                    pass_at_k = auto_test.cal_pass_at_k(
                        n, k, evaluated_results[task][test_class]['success'] + evaluated_results[task][test_class]['partial_success'])
                    partial_success_num += pass_at_k
                sum_num += 1
            except:
                # Class-level metrics (except block in cal_metrics_pass_at_k)
                if evaluated_results[task][test_class]['class_success'] != 0:
                    # Use the actual cal_pass_at_k function
                    pass_at_k = auto_test.cal_pass_at_k(
                        n, k, evaluated_results[task][test_class]['class_success'])
                    class_success_num += pass_at_k
                k_success = evaluated_results[task][test_class]['class_success'] + \
                    evaluated_results[task][test_class]['class_partial_success']
                if k_success != 0:
                    # Use the actual cal_pass_at_k function
                    pass_at_k = auto_test.cal_pass_at_k(n, k, k_success)
                    partial_success_class_num += pass_at_k
    
    return {
        'fun_success': success_num / sum_num if sum_num > 0 else 0,
        'class_success': class_success_num / class_num if class_num > 0 else 0,
        'fun_partial_success': partial_success_num / sum_num if sum_num > 0 else 0,
        'class_partial_success': partial_success_class_num / class_num if class_num > 0 else 0
    }


def _evaluate_prompt_results(model_result, auto_test):
    """
    Process test results using the EXACT same logic as test_pipeline.evaluate().
    This ensures consistency in how results are categorized.
    
    Args:
        model_result: dict with structure {task_id: {test_code_name: {test_class: result_item}}}
        auto_test: AutoTest instance for accessing get_test_answer method
    
    Returns:
        dict: Evaluated results in the same format as test_pipeline.evaluate()
    """
    result_dict = {}
    
    for task in model_result:
        result_dict[task] = {}
        for test_num in model_result[task]:
            temp_result = {"success": 0, "partial_success": 0, "fail": 0, "error": 0}
            
            for test_class in model_result[task][test_num]:
                if test_class not in result_dict[task]:
                    result_dict[task][test_class] = {}
                    result_dict[task]["TestClass"] = {}
                    result_dict[task]["TestClass"]["ClassEachTestResult"] = []
                    result_dict[task][test_class]['success'] = 0
                    result_dict[task][test_class]['partial_success'] = 0
                    result_dict[task][test_class]['fail'] = 0
                    result_dict[task][test_class]['error'] = 0
                    result_dict[task][test_class]["EachTestResult"] = []
                    result_dict[task]["TestClass"]["class_success"] = 0
                    result_dict[task]["TestClass"]["class_partial_success"] = 0
                    result_dict[task]["TestClass"]["class_fail"] = 0
                
                test_answer = auto_test.get_test_answer(model_result[task][test_num][test_class])
                result_dict[task][test_class][test_answer] += 1
                result_dict[task][test_class]["EachTestResult"].append(test_answer)
                temp_result[test_answer] += 1
            
            # Class-level success determination (same logic as test_pipeline)
            if temp_result['success'] == len(model_result[task][test_num]):
                result_dict[task]["TestClass"]["class_success"] += 1
                result_dict[task]["TestClass"]["ClassEachTestResult"].append("class_success")
            elif temp_result['fail'] == 0 and temp_result['error'] == 0:
                result_dict[task]["TestClass"]["class_partial_success"] += 1
                result_dict[task]["TestClass"]["ClassEachTestResult"].append("class_partial_success")
            else:
                result_dict[task]["TestClass"]["class_fail"] += 1
                result_dict[task]["TestClass"]["ClassEachTestResult"].append("class_fail")
    
    return result_dict

def _cleanup_pycache(directory):
    """Clean up __pycache__ directories and .pyc files"""
    for root, dirs, files in os.walk(directory):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
            except:
                pass
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
                except:
                    pass

def _cleanup_test_files(directory, prompt_id):
    """Clean up test files for a specific prompt_id"""
    for file in os.listdir(directory):
        if file.endswith('.py') and f'_prompt_{prompt_id}' in file:
            try:
                os.remove(os.path.join(directory, file))
            except:
                pass

def _cleanup_all_test_files(directory):
    """Clean up all remaining test files (ClassEval_*.py)"""
    for file in os.listdir(directory):
        if file.endswith('.py') and file.startswith('ClassEval_'):
            try:
                os.remove(os.path.join(directory, file))
            except:
                    pass


def convert_function_only_to_method_level(class_level_output_path, method_level_output_path, eval_data_path):
    """
    Convert function-only/full-context class-level output to method-level output.
    Each method is combined with ground truth methods for individual evaluation.
    
    Args:
        class_level_output_path: Path to class-level output JSON file (generated by function-only/full-context)
        method_level_output_path: Path to save method-level output JSONL file
        eval_data_path: Path to evaluation data (ClassEval_data.json) containing ground truth
    """
    # Load class-level predictions
    with open(class_level_output_path, 'r', encoding='utf-8') as f:
        class_predictions = json.load(f)
    
    # Load evaluation data (ground truth)
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # Create lookup for eval_data by task_id
    eval_data_dict = {item['task_id']: item for item in eval_data}
    
    method_records = []
    
    print(f"\n{'='*60}")
    print("Converting class-level to method-level output...")
    print(f"{'='*60}\n")
    
    for task_pred in tqdm(class_predictions, desc="Converting tasks"):
        task_id = task_pred['task_id']
        class_completion = task_pred.get('completion', '')
        
        if task_id not in eval_data_dict:
            print(f"⚠️  Task {task_id} not found in eval data, skipping...")
            continue
        
        task_eval = eval_data_dict[task_id]
        methods_info = task_eval.get('methods_info', [])
        imports = task_eval.get('import_statement', [])
        class_constructor = task_eval.get('class_constructor', '')
        
        if not methods_info:
            print(f"⚠️  No methods_info for {task_id}, skipping...")
            continue
        
        # Extract generated methods from class completion
        generated_methods = _extract_all_methods_from_class(class_completion)
        
        # For each method, create a record with generated method + ground truth other methods
        for method_info in methods_info:
            method_name = method_info['method_name']
            
            # Build class code: imports + constructor + methods
            class_code = '\n'.join(imports) + '\n' + class_constructor
            
            # Add all methods
            for m_info in methods_info:
                m_name = m_info['method_name']
                
                if m_name == method_name:
                    # Use generated method
                    if method_name in generated_methods:
                        class_code += '\n\n' + generated_methods[method_name]
                    else:
                        # Fallback: use signature + pass if generation failed
                        print(f"⚠️  Generated method {method_name} not found in {task_id}, using fallback")
                        method_sig = _get_method_signature_from_description(
                            m_info['method_description'], method_name
                        )
                        class_code += '\n\n' + method_sig + '\n        pass'
                else:
                    # Use ground truth method
                    if 'solution_code' in m_info:
                        class_code += '\n\n' + m_info['solution_code']
                    else:
                        print(f"⚠️  No solution_code for {m_name} in {task_id}")
                        method_sig = _get_method_signature_from_description(
                            m_info['method_description'], m_name
                        )
                        class_code += '\n\n' + method_sig + '\n        pass'
            
            # Create method-level record
            method_record = {
                'task_id': task_id,
                'class_name': task_eval['class_name'],
                'method_name': method_name,
                'prediction': generated_methods.get(method_name, ''),
                'class_code': class_code
            }
            
            method_records.append(method_record)
    
    # Save to JSONL file
    with open(method_level_output_path, 'w', encoding='utf-8') as f:
        for record in method_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Converted {len(method_records)} method records")
    print(f"📁 Saved to: {method_level_output_path}")
    
    return method_records


def _extract_all_methods_from_class(class_code):
    """
    Extract all methods from a complete class code.
    Returns a dictionary mapping method_name -> method_code (with proper indentation).
    """
    methods = {}
    
    # Split by lines
    lines = class_code.split('\n')
    
    current_method_name = None
    current_method_lines = []
    in_method = False
    method_indent = None
    
    for line in lines:
        # Check if this is a method definition line
        method_match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
        
        if method_match:
            # Save previous method if exists
            if current_method_name and current_method_lines:
                methods[current_method_name] = '\n'.join(current_method_lines)
            
            # Start new method
            method_indent = len(method_match.group(1))
            current_method_name = method_match.group(2)
            current_method_lines = [line]
            in_method = True
        elif in_method:
            # Check if we're still in the method (indentation or blank line)
            if line.strip() == '':
                # Blank line - keep it if we're in a method
                current_method_lines.append(line)
            elif line.startswith(' ' * (method_indent + 1)) or line.strip().startswith('@'):
                # Indented more than method definition or decorator
                current_method_lines.append(line)
            else:
                # End of method (new def at same level or less indentation)
                if current_method_name and current_method_lines:
                    methods[current_method_name] = '\n'.join(current_method_lines)
                current_method_name = None
                current_method_lines = []
                in_method = False
    
    # Save last method
    if current_method_name and current_method_lines:
        methods[current_method_name] = '\n'.join(current_method_lines)
    
    return methods