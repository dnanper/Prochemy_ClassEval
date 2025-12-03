import os
import re
import random
import subprocess
import concurrent.futures
from tqdm import tqdm
import json

TASK_DESCRIBER = """
You are an expert prompt engineer.
"""

INFORMATION = """
Please help me improve the given prompt to get a more helpful and harmless response.
Suppose I need to generate a Python program based on natural language descriptions.
The generated Python program should be able to complete the tasks described in natural language and pass any test cases specific to those tasks.\n
"""

FORMAT = """
You may add any information you think will help improve the task's effectiveness during the prompt optimization process.
If you find certain expressions and wording in the original prompt inappropriate, you can also modify these usages.
Ensure that the optimized prompt includes a detailed task description and clear process guidance added to the original prompt.
Wrap the optimized prompt in {{}}.
"""

def GEN_ANSWER(prompt, model_obj, tokenizer_or_name, model_type="openai", use_chat_template = False):
    if isinstance(tokenizer_or_name, str):
        if model_type == "gemini":
            # Gemini API
            full_prompt = f"{TASK_DESCRIBER}\n\n{INFORMATION + prompt + FORMAT}"
            try:
                response = model_obj.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 2500,
                    }
                )
                print("RESPONSE: ", response)
                if response.candidates and response.candidates[0].content.parts:
                    # print("here:: ", response.text)
                    return response.text
                else:
                    parts_text = "".join([part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')])
                    # print("here-2:: ", parts_text)
                    return parts_text if parts_text else ""
                print("=======================")
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

        # model = model_obj
        # tokenizer = tokenizer_or_name
        # # need update for QwenBase/Instruct - CodeLlama chat_template/instruction
        # if use_chat_template:
        #     messages = [
        #         {"role": "system", "content": task_describe},
        #         {"role": "user", "content": information + prompt + format}
        #     ]
        #     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # else:
        #     text =  f"### Instruction:\n{task_describe}\n\n{information + prompt + format}\n### Response:\n{{"
        # inputs = tokenizer(
        #     text, 
        #     return_tensors="pt", 
        #     padding=True,
        #     truncation=True,
        #     # max_length=4096
        # ).to(model.device)
        
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=500,
        #         temperature=0.7,
        #         do_sample=True,
        #         top_p=0.9, 
        #         pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        #         eos_token_id=tokenizer.eos_token_id,
        #     )
        # full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("full text: ", full_text)
        # generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        # response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # return "{{" + response

def extract_wrapped_content(text):
    match = re.search(r'\{\{(.*?)\}\}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text

def process_optimization_task(task_id, prompt, model_obj, tokenizer_or_name, model_type="openai"):
    attempts = 0
    while attempts < 5:
        completion = GEN_ANSWER(prompt, model_obj, tokenizer_or_name, model_type, use_chat_template = False)
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

def generate_new_prompts(existing_prompts, model_obj, tokenizer_or_name, model_type="openai", num_new_prompts=10):
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
            
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
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

def optimize_prompts(input_file, output_file, model_obj, tokenizer_or_name, model_type="openai", num_new_prompts=10):
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