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

from utils import read_json, write_jsonl

def GEN_SOLUTION(task_describe, prompts, model_obj, tokenizer_or_name, model_type = "openai", batch_size=8):
    results = []
    if isinstance(tokenizer_or_name, str):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for prompt in prompts:
                future = executor.submit(
                    _gen_api,
                    task_describe,
                    prompt,
                    model_obj,
                    tokenizer_or_name,
                    model_type
                )
                futures.append(future)
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="API Batch Processing"):
                results.append(future.result())
    else:
        print("Not import so skipping HF model generation.")

        # CHAT TEMPLATE USE/NOT USE CONFIG HERE
        # model = model_obj
        # tokenizer = tokenizer_or_name
        # for i in tqdm(range(0, len(prompts), batch_size), desc="HF Batch Processing"):
        #     batch_prompts = prompts[i:i + batch_size]
        #     batch_results = _gen_hf(
        #         task_describe,
        #         batch_prompts,
        #         model,
        #         tokenizer,
        #         use_chat_template = False
        #     )
            # results.extend(batch_results)
    return results
# helper function

def _gen_api(task_describe, prompt, model_obj, model_name, model_type = "openai"):
    """Generate completion API"""
    attempts = 0
    response_text = ""
    instruction = f"Please complete the class {prompt['class_name']} in the following code.\n{prompt['skeleton']}"
    while attempts < 3:
        try:
            if model_type == "gemini":
                # Gemini API with safety settings to reduce blocking
                import google.generativeai as genai
                full_prompt = f"{task_describe}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
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

# helper function
# def _gen_hf(task_describe, prompts, model, tokenizer, use_chat_template = False):
#     """Generate batch with HuggingFace"""
#     if use_chat_template:
#         all_messages = [
#             [
#                 {"role": "system", "content": task_describe},
#                 {"role": "user", "content": prompt}
#             ]
#             for prompt in prompts
#         ]
#         texts = [
#             tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
#             for msgs in all_messages
#         ]
#     else:
#         texts = [
#             f"### Instruction:\n{task_describe}\n\n{prompt}\n### Response:\n```python"
#             for prompt in prompts
#         ]
#     inputs = tokenizer(
#         texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=4096
#     ).to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=1000,
#             temperature=0.0,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     results = []
#     for i, output in enumerate(outputs):
#         generated_ids = output[inputs['input_ids'].shape[1]:]
#         full_text = tokenizer.decode(output, skip_special_tokens=True)
#         # print("Full Text: ", full_text)
#         response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         # print("Response Text: ", response_text)
#         match = re.search(r'```(?:python)?(.*?)```', full_text, re.DOTALL)
#         if match:
#             results.append(match.group(1).strip())
#         else:
#             results.append(response_text)
#     return results

# Generate solutions and save to the specified directory
def generate_solutions(test_set_path, mutated_prompt_path, output_directory, model_obj, tokenizer_or_name, model_type = "openai", batch_size=8):
    os.makedirs(output_directory, exist_ok=True)
    
    # Load mutated prompts - support both .jsonl and .json formats
    if mutated_prompt_path.endswith('.jsonl'):
        from utils import read_jsonl
        mutated_prompts = {
            task["prompt_id"]: task["mutated_prompt"] 
            for task in read_jsonl(mutated_prompt_path)
        }
    else:
        mutated_prompts = {
            task["prompt_id"]: task["mutated_prompt"] 
            for task in read_json(mutated_prompt_path)
        }
    
    all_tasks = list(read_json(test_set_path))
    is_hf_model = not isinstance(tokenizer_or_name, str)
    # log
    print(f"Processing mode: {'HuggingFace (GPU Batch)' if is_hf_model else 'API (Parallel)'}")
    print(f"Batch size: {batch_size if is_hf_model else 'N/A (using 10 parallel workers)'}")
    
    for prompt_id, task_describe in mutated_prompts.items():
        print(f"\n{'='*60}")
        print(f"Processing prompt_id: {prompt_id}")
        print(f"{'='*60}")
        task_prompts = [{'class_name': task["class_name"], 'skeleton': task["skeleton"]} for task in all_tasks]
        
        completions = GEN_SOLUTION(
            task_describe,
            task_prompts,
            model_obj,
            tokenizer_or_name,
            model_type,
            batch_size=batch_size,
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
    This function follows the methodology from test_pipeline.py in ClassEval.
    
    Args:
        output_folder: folder containing generated solution files (train_set_{model}_{prompt_id}.jsonl)
        train_set_path: path to train_set.json (format similar to ClassEval_data.json)
        prompts_file_path: path to mutated_prompts file
        best_prompt_output_path: path to save the best prompts
    """
    # Load train_set data
    train_set_data = {}
    with open(train_set_path, encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        train_set_data[item['task_id']] = item
    
    # Dictionary to store results for each prompt_id
    prompt_results = {}
    
    print(f"\n{'='*60}")
    print("Starting evaluation of generated solutions...")
    print(f"{'='*60}\n")
    
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
        completions = {}
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    completions[item['task_id']] = item['completion']
                except json.JSONDecodeError:
                    continue
        
        # Run tests for each task
        task_results = {}
        
        for task_id, completion in tqdm(completions.items(), desc=f"Testing prompt {prompt_id}"):
            if task_id not in train_set_data:
                print(f"Warning: task_id {task_id} not found in train_set, skipping...")
                continue
            
            task_data = train_set_data[task_id]
            
            # Generate test file

            ##### TEST
            print(completion)
            # completion = task_data['solution_code']
            #####
            print("______________________________________________")
            # print(completion)

            # completion = completion.replace('```python', '').replace('```', '')

            test_code = '\n'.join(task_data['import_statement']) + '\n' + completion + '\n' + task_data['test']
            test_file_name = f"{task_id}_prompt_{prompt_id}"
            test_file_path = os.path.join(output_folder, f"{test_file_name}.py")
            
            try:
                # Write test file
                with open(test_file_path, 'w', encoding='utf-8') as f:
                    f.write(test_code)
                
                # Add output_folder to sys.path for importing
                if output_folder not in sys.path:
                    sys.path.insert(0, output_folder)
                
                # Run unit tests
                test_result = _run_unit_test_safe(test_file_name, task_data['test_classes'], timeout=5)
                task_results[task_id] = test_result
                
            except Exception as e:
                print(f"Error testing {task_id}: {e}")
                task_results[task_id] = {'success': 0, 'partial_success': 0, 'fail': 0, 'error': 1}
            
            finally:
                # Clean up test file
                if os.path.exists(test_file_path):
                    try:
                        os.remove(test_file_path)
                    except:
                        pass
        
        # Calculate metrics for this prompt
        prompt_results[prompt_id] = _calculate_prompt_metrics(task_results)
        
        print(f"\nPrompt {prompt_id} Results:")
        print(f"  Function Success Rate: {prompt_results[prompt_id]['fun_success_rate']:.2%}")
        print(f"  Class Success Rate: {prompt_results[prompt_id]['class_success_rate']:.2%}")
        print(f"  Total Success: {prompt_results[prompt_id]['total_success']}/{prompt_results[prompt_id]['total_tests']}")
    
    # Clean up any remaining Python cache
    _cleanup_pycache(output_folder)
    
    # Select best prompts based on combined metric
    if not prompt_results:
        print("\nNo valid results found!")
        return
    
    # Calculate combined score (weighted average of function and class success)
    for prompt_id in prompt_results:
        fun_rate = prompt_results[prompt_id]['fun_success_rate']
        class_rate = prompt_results[prompt_id]['class_success_rate']
        # Weight class success more heavily (0.6) than function success (0.4)
        prompt_results[prompt_id]['combined_score'] = 0.4 * fun_rate + 0.6 * class_rate
    
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
        print(f"  Function Success Rate: {r['fun_success_rate']:.2%}")
        print(f"  Class Success Rate: {r['class_success_rate']:.2%}")
        print(f"  Combined Score: {r['combined_score']:.4f}")
        print(f"  Total Tests: {r['total_success']}/{r['total_tests']}")
    
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


def _run_unit_test_safe(test_module_name, test_classes, timeout=5):
    """
    Safely run unit tests with timeout protection.
    Returns aggregated results across all test classes.
    """
    @func_set_timeout(timeout)
    def _run_test():
        try:
            module = importlib.import_module(test_module_name)
            
            total_errors = 0
            total_failures = 0
            total_tests_run = 0
            
            for test_class in test_classes:
                try:
                    test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_class))
                    test_result = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(test_suite)
                    
                    total_errors += len(test_result.errors)
                    total_failures += len(test_result.failures)
                    total_tests_run += test_result.testsRun
                except Exception as e:
                    # If a test class fails to load/run, treat all its tests as errors
                    pass
            
            return {
                'errors': total_errors,
                'failures': total_failures,
                'testsRun': total_tests_run
            }
        except Exception as e:
            return {'errors': 0, 'failures': 0, 'testsRun': 0}
    
    try:
        test_result = _run_test()
        
        # Categorize result
        if test_result['testsRun'] == 0 or test_result['errors'] == test_result['testsRun']:
            return {'success': 0, 'partial_success': 0, 'fail': 0, 'error': 1}
        elif test_result['errors'] + test_result['failures'] == 0:
            return {'success': 1, 'partial_success': 0, 'fail': 0, 'error': 0}
        elif test_result['errors'] + test_result['failures'] < test_result['testsRun']:
            return {'success': 0, 'partial_success': 1, 'fail': 0, 'error': 0}
        else:
            return {'success': 0, 'partial_success': 0, 'fail': 1, 'error': 0}
    except:
        return {'success': 0, 'partial_success': 0, 'fail': 0, 'error': 1}


def _calculate_prompt_metrics(task_results):
    """
    Calculate aggregated metrics for a prompt based on task results.
    Similar to the metrics calculation in test_pipeline.py
    """
    total_tests = len(task_results)
    success_count = sum(1 for r in task_results.values() if r['success'] > 0)
    partial_success_count = sum(1 for r in task_results.values() if r['partial_success'] > 0)
    
    # Function-level metrics: count of successful functions
    fun_success = success_count
    fun_partial = success_count + partial_success_count
    
    # Class-level metrics: a class is successful if all its tests pass
    class_success = success_count
    class_partial = success_count + partial_success_count
    
    return {
        'fun_success_rate': fun_success / total_tests if total_tests > 0 else 0,
        'class_success_rate': class_success / total_tests if total_tests > 0 else 0,
        'fun_partial_rate': fun_partial / total_tests if total_tests > 0 else 0,
        'class_partial_rate': class_partial / total_tests if total_tests > 0 else 0,
        'total_success': success_count,
        'total_partial': partial_success_count,
        'total_tests': total_tests
    }


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