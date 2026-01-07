import os
import re
import random
import subprocess
import concurrent.futures
from tqdm import tqdm
import json

# ============================================================================
# SYSTEM PROMPTS FOR MUTATION
# ============================================================================

TASK_DESCRIBER = """
You are an expert prompt engineer specializing in optimizing prompts for code generation tasks.
"""

INFORMATION = """
Please help me improve the given prompt to get a more helpful and accurate response.
Suppose I need to generate a Python program based on natural language descriptions.
The generated Python program should be able to complete the tasks described in natural language and pass any test cases specific to those tasks.
"""

FORMAT = """
You may add any information you think will help improve the task's effectiveness during the prompt optimization process.
If you find certain expressions and wording in the original prompt inappropriate, you can also modify these usages.
Ensure that the optimized prompt includes a detailed task description and clear process guidance added to the original prompt.
Wrap the optimized prompt in {{}}.
"""

# ============================================================================
# FEEDBACK-GUIDED MUTATION PROMPTS (Level 1)
# ============================================================================

FEEDBACK_GUIDED_TASK_DESCRIBER = """
You are an expert prompt engineer specializing in optimizing prompts for code generation tasks.
You will be given a prompt along with its PERFORMANCE FEEDBACK from evaluation.
Use this feedback to make targeted improvements to the prompt.
"""

FEEDBACK_GUIDED_INFORMATION = """
I need to improve a prompt that is used for Python code generation.
The prompt was evaluated on a test set and I have detailed performance feedback.

Your task:
1. Analyze the performance feedback to understand the weaknesses
2. Identify patterns in the errors (e.g., edge cases, type handling, logic errors)
3. Modify the prompt to specifically address these weaknesses
4. Keep the strengths of the original prompt while fixing the issues
"""

FEEDBACK_GUIDED_FORMAT = """
Based on the feedback, improve the prompt by:
- Adding specific instructions to handle common error patterns
- Including guidance for edge cases if they are causing failures
- Clarifying any ambiguous instructions that may lead to incorrect code
- Adding examples or hints that could help avoid the observed errors

Wrap the optimized prompt in {{}}.
Do NOT include the performance feedback in your optimized prompt - it should be a clean, improved prompt.
"""


def GEN_ANSWER(prompt, model_obj, tokenizer_or_name, model_type="openai", use_chat_template=False):
    """Original mutation without feedback."""
    if isinstance(tokenizer_or_name, str):
        if model_type == "gemini":
            # Gemini API
            full_prompt = f"{TASK_DESCRIBER}\n\n{INFORMATION + prompt + FORMAT}"
            try:
                response = model_obj.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 5000,
                    }
                )
                print("RESPONSE: ", response)
                if response.candidates and response.candidates[0].content.parts:
                    return response.text
                else:
                    parts_text = "".join([part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')])
                    return parts_text if parts_text else ""
            except Exception as e:
                print(f"Gemini API error: {e}")
                return ""
        else:
            # OpenAI API
            response = model_obj.chat.completions.create(
                model=tokenizer_or_name,
                messages=[
                    {"role": "system", "content": TASK_DESCRIBER},
                    {"role": "user", "content": INFORMATION + prompt + FORMAT}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
    else:
        print("No import so not needed HF model generation code.")
        return ""


def GEN_ANSWER_WITH_FEEDBACK(prompt_text, feedback_summary, model_obj, tokenizer_or_name, model_type="openai"):
    """
    Feedback-guided mutation: Generate improved prompt using performance feedback.
    
    Args:
        prompt_text: The original prompt to improve
        feedback_summary: Human-readable feedback from evaluation
        model_obj: LLM model object
        tokenizer_or_name: Tokenizer or model name
        model_type: Type of model ("openai", "gemini", "hf")
    
    Returns:
        str: Generated improved prompt
    """
    # Construct the user message with prompt and feedback
    user_message = f"""
=== ORIGINAL PROMPT ===
{prompt_text}

=== PERFORMANCE FEEDBACK ===
{feedback_summary}

=== YOUR TASK ===
{FEEDBACK_GUIDED_INFORMATION}

{FEEDBACK_GUIDED_FORMAT}
"""
    
    if isinstance(tokenizer_or_name, str):
        if model_type == "gemini":
            # Gemini API
            full_prompt = f"{FEEDBACK_GUIDED_TASK_DESCRIBER}\n\n{user_message}"
            try:
                response = model_obj.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 5000,
                    }
                )
                print("[Feedback-Guided] RESPONSE received")
                if response.candidates and response.candidates[0].content.parts:
                    return response.text
                else:
                    parts_text = "".join([part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')])
                    return parts_text if parts_text else ""
            except Exception as e:
                print(f"Gemini API error: {e}")
                return ""
        else:
            # OpenAI API
            response = model_obj.chat.completions.create(
                model=tokenizer_or_name,
                messages=[
                    {"role": "system", "content": FEEDBACK_GUIDED_TASK_DESCRIBER},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
    else:
        print("HF model feedback-guided mutation not implemented yet.")
        return ""


def extract_wrapped_content(text):
    """Extract content wrapped in {{}}."""
    match = re.search(r'\{\{(.*?)\}\}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text


def process_optimization_task(task_id, prompt, model_obj, tokenizer_or_name, model_type="openai"):
    """Original mutation task processor (without feedback)."""
    attempts = 0
    while attempts < 5:
        completion = GEN_ANSWER(prompt, model_obj, tokenizer_or_name, model_type, use_chat_template=False)
        if not completion:
            print(f"GEN_ANSWER: Task {task_id}: Empty completion on attempt {attempts + 1}")
            attempts += 1
            continue
        wrapped_content = extract_wrapped_content(completion)
        if wrapped_content:
            return dict(prompt_id=task_id, mutated_prompt=wrapped_content)
        else:
            print(f"GEN_ANSWER: Task {task_id}: No wrapped content found. Retrying...")
        attempts += 1
    print(f"GEN_ANSWER: Task {task_id}: Failed after 5 attempts. Returning original prompt.")
    return dict(prompt_id=task_id, mutated_prompt=prompt)


def process_optimization_task_with_feedback(task_id, prompt_text, feedback_summary, model_obj, tokenizer_or_name, model_type="openai"):
    """
    Feedback-guided mutation task processor.
    
    Args:
        task_id: ID for the new prompt
        prompt_text: Original prompt text to improve
        feedback_summary: Performance feedback from evaluation
        model_obj: LLM model object
        tokenizer_or_name: Tokenizer or model name
        model_type: Type of model
    
    Returns:
        dict: {prompt_id, mutated_prompt}
    """
    attempts = 0
    while attempts < 5:
        completion = GEN_ANSWER_WITH_FEEDBACK(
            prompt_text, 
            feedback_summary, 
            model_obj, 
            tokenizer_or_name, 
            model_type
        )
        if not completion:
            print(f"[Feedback-Guided] Task {task_id}: Empty completion on attempt {attempts + 1}")
            attempts += 1
            continue
        wrapped_content = extract_wrapped_content(completion)
        if wrapped_content:
            return dict(prompt_id=task_id, mutated_prompt=wrapped_content)
        else:
            print(f"[Feedback-Guided] Task {task_id}: No wrapped content found. Retrying...")
        attempts += 1
    print(f"[Feedback-Guided] Task {task_id}: Failed after 5 attempts. Returning original prompt.")
    return dict(prompt_id=task_id, mutated_prompt=prompt_text)


def generate_new_prompts(existing_prompts, model_obj, tokenizer_or_name, model_type="openai", num_new_prompts=10):
    """Original prompt generation without feedback."""
    new_prompts = []
    is_hf_model = not isinstance(tokenizer_or_name, str)

    if is_hf_model:
        print("[HF Model] Processing prompt optimization sequentially...")
        for task_id in tqdm(range(num_new_prompts), desc="Optimizing prompts"):
            random_prompt = random.choice(existing_prompts)
            prompt_text = random_prompt['mutated_prompt']
            formatted_prompt = f"The prompt ready to be optimized are as follows and wrapped in []:\n[{prompt_text}]\n"
            
            result = process_optimization_task(
                task_id, 
                formatted_prompt, 
                model_obj, 
                tokenizer_or_name
            )
            new_prompts.append(result)
    else:
        print("[API Model] Processing prompt optimization in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task_id in range(num_new_prompts):
                random_prompt = random.choice(existing_prompts)
                prompt_text = random_prompt['mutated_prompt']
                formatted_prompt = f"The prompt ready to be optimized are as follows and wrapped in []:\n[{prompt_text}]\n"
                futures.append(executor.submit(process_optimization_task, task_id, formatted_prompt, model_obj, tokenizer_or_name, model_type))
        
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks"):
                new_prompts.append(future.result())

    return new_prompts


def generate_new_prompts_with_feedback(feedback_data, model_obj, tokenizer_or_name, model_type="openai", num_new_prompts=10, selection_strategy="top_k"):
    """
    Generate new prompts using feedback-guided mutation.
    
    Args:
        feedback_data: List of dicts with {prompt_id, prompt_text, feedback_summary, combined_score, ...}
        model_obj: LLM model object
        tokenizer_or_name: Tokenizer or model name
        model_type: Type of model
        num_new_prompts: Number of new prompts to generate
        selection_strategy: How to select prompts for mutation
            - "top_k": Mutate the top-k best prompts (default)
            - "proportional": Probability proportional to score
            - "all": Mutate all prompts in feedback_data
    
    Returns:
        list: List of new prompt dicts
    """
    new_prompts = []
    is_hf_model = not isinstance(tokenizer_or_name, str)
    
    print(f"\n{'='*60}")
    print(f"[Feedback-Guided Mutation] Starting with {len(feedback_data)} prompts")
    print(f"Selection strategy: {selection_strategy}")
    print(f"Target: {num_new_prompts} new prompts")
    print(f"{'='*60}\n")
    
    # Select prompts for mutation based on strategy
    selected_prompts = []
    
    if selection_strategy == "top_k":
        # Use top-k prompts (already sorted by score in feedback_data)
        k = min(len(feedback_data), max(1, num_new_prompts // 2))  # At least 1, at most half of target
        selected_prompts = feedback_data[:k]
        print(f"Selected top {k} prompts for mutation")
        
    elif selection_strategy == "proportional":
        # Weighted random selection based on score
        scores = [max(0.01, p.get('combined_score', 0.01)) for p in feedback_data]
        total_score = sum(scores)
        probabilities = [s / total_score for s in scores]
        
        selected_indices = random.choices(range(len(feedback_data)), weights=probabilities, k=num_new_prompts)
        selected_prompts = [feedback_data[i] for i in selected_indices]
        print(f"Selected {len(selected_prompts)} prompts using proportional sampling")
        
    elif selection_strategy == "all":
        # Mutate all prompts in feedback_data
        selected_prompts = feedback_data
        print(f"Using all {len(selected_prompts)} prompts for mutation")
    else:
        # Default to top-k
        selected_prompts = feedback_data[:min(len(feedback_data), num_new_prompts)]
    
    # Distribute mutations across selected prompts
    mutations_per_prompt = max(1, num_new_prompts // len(selected_prompts))
    remaining = num_new_prompts - (mutations_per_prompt * len(selected_prompts))
    
    # Build task list
    mutation_tasks = []
    task_counter = 0
    
    for i, prompt_info in enumerate(selected_prompts):
        # How many mutations for this prompt
        n_mutations = mutations_per_prompt + (1 if i < remaining else 0)
        
        for _ in range(n_mutations):
            mutation_tasks.append({
                'task_id': task_counter,
                'prompt_text': prompt_info.get('prompt_text', ''),
                'feedback_summary': prompt_info.get('feedback_summary', ''),
                'original_prompt_id': prompt_info.get('prompt_id', 0),
                'original_score': prompt_info.get('combined_score', 0)
            })
            task_counter += 1
    
    print(f"Created {len(mutation_tasks)} mutation tasks")
    
    # Execute mutations
    if is_hf_model:
        print("[HF Model] Processing feedback-guided mutation sequentially...")
        for task in tqdm(mutation_tasks, desc="Feedback-guided mutation"):
            result = process_optimization_task_with_feedback(
                task['task_id'],
                task['prompt_text'],
                task['feedback_summary'],
                model_obj,
                tokenizer_or_name
            )
            new_prompts.append(result)
    else:
        print("[API Model] Processing feedback-guided mutation in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task in mutation_tasks:
                future = executor.submit(
                    process_optimization_task_with_feedback,
                    task['task_id'],
                    task['prompt_text'],
                    task['feedback_summary'],
                    model_obj,
                    tokenizer_or_name,
                    model_type
                )
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing feedback-guided mutations"):
                new_prompts.append(future.result())
    
    return new_prompts


def optimize_prompts(input_file, output_file, model_obj, tokenizer_or_name, model_type="openai", num_new_prompts=10):
    """Original optimize_prompts function (without feedback)."""
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    with open(input_file, 'r') as file:
        prompts = [json.loads(line) for line in file]
    
    print(f"Loaded {len(prompts)} existing prompts from {input_file}")

    new_prompts = generate_new_prompts(prompts, model_obj, tokenizer_or_name, model_type, num_new_prompts)

    existing_ids = {prompt['prompt_id'] for prompt in prompts}
    new_id = max(existing_ids) + 1 if existing_ids else 0
    
    for new_prompt in new_prompts:
        while new_id in existing_ids:
            new_id += 1
        new_prompt['prompt_id'] = new_id
        existing_ids.add(new_id)
        new_id += 1

    combined_prompts = prompts + new_prompts
    with open(output_file, 'w') as out_file:
        for prompt in combined_prompts:
            json.dump(prompt, out_file)
            out_file.write('\n')

    print(f"Saved {len(combined_prompts)} prompts ({len(prompts)} old + {len(new_prompts)} new) to {output_file}")


def optimize_prompts_with_feedback(feedback_file, best_prompts_file, output_file, model_obj, tokenizer_or_name, model_type="openai", num_new_prompts=10, selection_strategy="top_k"):
    """
    Feedback-guided prompt optimization.
    
    Args:
        feedback_file: Path to mutation_feedback.jsonl (from evaluate_and_select_best_prompts with collect_feedback=True)
        best_prompts_file: Path to best_prompts.jsonl (for combining with new prompts)
        output_file: Path to save combined prompts
        model_obj: LLM model object
        tokenizer_or_name: Tokenizer or model name
        model_type: Type of model
        num_new_prompts: Number of new prompts to generate
        selection_strategy: How to select prompts for mutation ("top_k", "proportional", "all")
    """
    # Load feedback data
    if not os.path.exists(feedback_file):
        print(f"[Warning] Feedback file {feedback_file} not found. Falling back to standard optimization.")
        return optimize_prompts(best_prompts_file, output_file, model_obj, tokenizer_or_name, model_type, num_new_prompts)
    
    feedback_data = []
    with open(feedback_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                feedback_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not feedback_data:
        print(f"[Warning] No feedback data loaded. Falling back to standard optimization.")
        return optimize_prompts(best_prompts_file, output_file, model_obj, tokenizer_or_name, model_type, num_new_prompts)
    
    print(f"\n{'='*60}")
    print(f"[Feedback-Guided Optimization]")
    print(f"Loaded {len(feedback_data)} prompts with feedback from {feedback_file}")
    print(f"{'='*60}\n")
    
    # Print feedback summary
    for i, fd in enumerate(feedback_data[:3]):  # Show top 3
        print(f"Prompt {fd.get('prompt_id', 'N/A')} (score: {fd.get('combined_score', 0):.4f}):")
        print(f"  Success Rate: {fd.get('success_rate', 0):.1%}")
        print(f"  Partial Success: {fd.get('partial_success_rate', 0):.1%}")
        raw_fb = fd.get('raw_feedback', {})
        top_errors = raw_fb.get('top_error_patterns', [])[:3]
        if top_errors:
            print(f"  Top Errors: {', '.join([f'{p}({c})' for p, c in top_errors])}")
        print()
    
    # Generate new prompts with feedback
    new_prompts = generate_new_prompts_with_feedback(
        feedback_data, 
        model_obj, 
        tokenizer_or_name, 
        model_type, 
        num_new_prompts,
        selection_strategy
    )
    
    # Load best prompts to combine
    best_prompts = []
    if os.path.exists(best_prompts_file):
        with open(best_prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    best_prompts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Assign new IDs to new prompts
    existing_ids = {p['prompt_id'] for p in best_prompts}
    new_id = max(existing_ids) + 1 if existing_ids else 0
    
    for new_prompt in new_prompts:
        while new_id in existing_ids:
            new_id += 1
        new_prompt['prompt_id'] = new_id
        existing_ids.add(new_id)
        new_id += 1
    
    # Combine and save
    combined_prompts = best_prompts + new_prompts
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for prompt in combined_prompts:
            json.dump(prompt, out_file, ensure_ascii=False)
            out_file.write('\n')
    
    print(f"\n{'='*60}")
    print(f"[Feedback-Guided Optimization Complete]")
    print(f"Saved {len(combined_prompts)} prompts ({len(best_prompts)} best + {len(new_prompts)} new) to {output_file}")
    print(f"{'='*60}\n")