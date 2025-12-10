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
    output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
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
        # print("Response Text: ", response_text)
        match = re.search(r'```(?:python)?(.*?)```', full_text, re.DOTALL)
        if match:
            results.append(match.group(1).strip())
        else:
            results.append(response_text)
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
    Similar to _gen_function_only but uses HF model instead of API.
    """
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


def _gen_full_context_hf(task_describe, prompt, model, tokenizer, use_chat_template=True):
    """
    Generate code with full context using HuggingFace model: provide full implementation of other methods,
    only hide the body of the target method to generate.
    This strategy gives maximum context to the model.
    """
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
                method_signature = _get_method_signature_from_description(
                    method['method_description'], 
                    method['method_name']
                )
                class_text += '\n\n' + method_signature
                class_text += '\n        """' + method['method_description'].split('"""')[1].split('"""')[0] + '"""'
                class_text += '\n        # TODO: Complete this method'
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
        
        # Construct prompt
        method_name = method_to_generate['method_name']
        inst = f"Please complete the method {method_name} in the following class {class_name}. The other methods are already implemented for your reference:\n\n"
        
        method_prompts.append(inst + class_text)
    
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
                'class_name': task["class_name"],
                'skeleton': task["skeleton"],
                'methods_info': task.get("methods_info", []),
                'import_statement': task.get("import_statement", []),
                'class_constructor': task.get("class_constructor", ""),
                'class_description': task.get("class_description", "")
            } for task in all_tasks]
        elif strategy == "function-only":
            # For function-only, need methods_info but no skeleton
            task_prompts = [{
                'class_name': task["class_name"],
                'methods_info': task.get("methods_info", []),
                'import_statement': task.get("import_statement", []),
                'class_constructor': task.get("class_constructor", ""),
                'class_description': task.get("class_description", "")
            } for task in all_tasks]
        elif strategy == "full-context":
            # For full-context, need methods_info with solution_code
            task_prompts = [{
                'class_name': task["class_name"],
                'methods_info': task.get("methods_info", []),
                'import_statement': task.get("import_statement", []),
                'class_constructor': task.get("class_constructor", ""),
                'class_description': task.get("class_description", "")
            } for task in all_tasks]
        else:
            # For holistic, just need class_name and skeleton
            task_prompts = [{'class_name': task["class_name"], 'skeleton': task["skeleton"]} for task in all_tasks]
        
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
        dummy = "dummy"
        for task, completion in zip(all_tasks, completions):
            # print("XXXXXXXXXXXXXXX")
            # print(task["prompt"])
            # print("<-->  Generated: ")
            # print(completion)
            if completion:
                samples.append({
                    'task_id': task['task_id'],
                    'completion': completion
                })
            else:
                print(f"No valid completion for task_id: {task['task_id']}")
                samples.append({
                    'task_id': task['task_id'],
                    'completion': dummy
                })
    
        # Create safe model name from model identifier
        if isinstance(tokenizer_or_name, str):
            model_name_safe = tokenizer_or_name.split('/')[-1].replace('.', '_').replace('-', '_')
        else:
            model_name_safe = "model"
        
        output_file = os.path.join(
            output_directory, 
            f"train_set_{model_name_safe}_{prompt_id}.jsonl"
        )
        write_jsonl(output_file, samples)
        print(f"Saved {len(samples)}/{len(all_tasks)} samples to {output_file}")

# Evaluate the generated solutions and select the best prompts
def evaluate_and_select_best_prompts(output_folder, train_set_path, prompts_file_path, best_prompt_output_path):
    """
    Evaluate generated solutions using unittest and select the best prompts based on test results.
    This function now uses AutoTest from test_pipeline.py for consistent evaluation logic.
    
    Args:
        output_folder: folder containing generated solution files (train_set_{model}_{prompt_id}.jsonl)
        train_set_path: path to train_set.json (format similar to ClassEval_data.json)
        prompts_file_path: path to mutated_prompts file
        best_prompt_output_path: path to save the best prompts
    """
    # Initialize AutoTest with train_set data
    # AutoTest expects filename without extension (PathUtil.eval_data adds .json)
    # Extract filename from path: "path/to/ClassEval_data.json" -> "ClassEval_data"
    train_set_name = os.path.splitext(os.path.basename(train_set_path))[0]
    auto_test = AutoTest(train_set_name)
    
    # Dictionary to store results for each prompt_id
    prompt_results = {}
    
    print(f"\n{'='*60}")
    print("Starting evaluation of generated solutions...")
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
                
            # Extract prompt_id from filename (e.g., train_set_model_123.jsonl -> 123)
            try:
                prompt_id = int(re.search(r'_(\d+)\.jsonl$', filename).group(1))
            except (IndexError, ValueError, AttributeError):
                print(f"Warning: Could not extract prompt_id from {filename}, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating Prompt ID: {prompt_id} ({filename})")
            print(f"{'='*60}")
            
            jsonl_file_path = os.path.join(output_folder, filename)
            
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
                            full_code = '\n'.join(auto_test.eval_data[task_id]['import_statement']) + '\n' + completion
                            code_list[task_id] = [full_code]  # List of 1 code per task
                    except json.JSONDecodeError:
                        continue
            
            # Generate test files for all tasks (reusing test_pipeline logic)
            for task_id in code_list:
                test_code = auto_test.eval_data[task_id]['test']
                task_code_list = code_list[task_id]
                auto_test.gen_py_file(f"{task_id}_prompt_{prompt_id}", task_code_list, test_code)
            
            # Run tests for each task (reusing test_pipeline logic)
            all_test_results = {}
            for task_id in tqdm(code_list, desc=f"Testing prompt {prompt_id}"):
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
            
            # Process results using the same logic as test_pipeline.evaluate()
            evaluated_results = _evaluate_prompt_results(all_test_results, auto_test)
            
            # Calculate metrics using the EXACT same logic as cal_metrics_pass_at_k
            # but inline to avoid file I/O issues
            prompt_results[prompt_id] = _calculate_metrics(evaluated_results, auto_test)
            
            print(f"\nPrompt {prompt_id} Results:")
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
    
    # Calculate combined score using formula: 0.4*fun_success + 0.6*class_success
    # Using the metrics directly from cal_metrics_pass_at_k()
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