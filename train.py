import os

from prompt_eval import generate_solutions, evaluate_and_select_best_prompts
from prompt_mutate import optimize_prompts

from openai import OpenAI
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

def create_model(model_name, model_type=None, role="main"):
    if model_type is None:
        if any(x in model_name.lower() for x in ["gpt", "deepseek"]):
            model_type = "openai"
        elif "gemini" in model_name.lower():
            model_type = "gemini"
        else:
            model_type = "hf"
    if model_type == "openai":
        print(f"[{role.upper()}] Using OpenAI API model: {model_name}")
        client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return client, model_name
    elif model_type == "gemini":
        print(f"[{role.upper()}] Using Gemini API model: {model_name}")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        client = genai.GenerativeModel(model_name)
        return client, model_name
    else:
        print(f"[{role.upper()}] Using Hugging Face model: {model_name}")

        # Not import so not needed
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer

BASE_DIR = os.getcwd()

TEST_SET_PATH = os.path.join(BASE_DIR, "ClassEval_data.json")

# file chứa thông tin như test cases, expected output, ... để đánh giá completion sinh từ GEN_SOLUTION.
PROBLEM_FILE_PATH = os.path.join(BASE_DIR, "ClassEval_data.json")
    
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "infer_7b_full_context")

initial_prompt_path = os.path.join(BASE_DIR, "best_prompt_7b_full_context.jsonl")

if not os.path.exists(initial_prompt_path):
    print(f"Error: Initial prompt file not found: {initial_prompt_path}")
    print("Please create an initial_prompts.jsonl file with seed prompts.")
    exit(1)
current_prompt_file = initial_prompt_path

MAIN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAIN_MODEL_TYPE = "hf"
# MUTATE_MODEL = "gpt-4o-mini"
# MUTATE_MODEL_TYPE = "openai"
# MUTATE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# MUTATE_MODEL_TYPE = "hf"
MUTATE_MODEL = "gemini-2.5-flash"
MUTATE_MODEL_TYPE = "gemini"

main_model, main_tokenizer = create_model(MAIN_MODEL, MAIN_MODEL_TYPE, "main")
mutate_model, mutate_tokenizer = create_model(MUTATE_MODEL, MUTATE_MODEL_TYPE, "mutate")

N_GENERATIONS = 1
N_NEW_PROMPTS = 5

for gen in range(N_GENERATIONS):
    print("\n" + "=" * 60)
    print(f"GENERATION {gen + 1}/{N_GENERATIONS}")
    print("=" * 60)

    # Step 1: Prompt Inference - Generate solutions
    print(f"\n[Step 1] Generating solutions using prompts from: {current_prompt_file}")
    output_directory = os.path.join(OUTPUT_BASE_DIR, f"gen_{gen}", "solutions")
    generate_solutions(
        test_set_path=TEST_SET_PATH,
        mutated_prompt_path=current_prompt_file,
        output_directory=output_directory,
        model_obj=main_model,
        tokenizer_or_name=main_tokenizer,
        model_type=MAIN_MODEL_TYPE,
        batch_size=1,
        strategy="full-context"  # Use "holistic" for entire class generation or "compositional" for method-by-method
    )

    # Step 2: Prompt Evaluation (Thru Completion) and Select best prompt
    print(f"\n[Step 2] Evaluating solutions in: {output_directory}")
    best_prompt_output_path = os.path.join(OUTPUT_BASE_DIR, f"gen_{gen}", "best_prompts.jsonl")
    evaluate_and_select_best_prompts(
        output_folder=output_directory,
        train_set_path=PROBLEM_FILE_PATH,
        prompts_file_path=current_prompt_file,
        best_prompt_output_path=best_prompt_output_path,
        strategy="full-context"
    )

    # Step 3: Prompt Mutation
    # if gen < N_GENERATIONS - 1:  # Don't optimize after last generation
    #     print(f"\n[Step 3] Optimizing prompts for next generation")
    #     next_prompt_file = os.path.join(OUTPUT_BASE_DIR, f"gen_{gen}", "optimized_prompts.jsonl")
    #     optimize_prompts(
    #         input_file=best_prompt_output_path,
    #         output_file=next_prompt_file,
    #         model_obj=mutate_model,
    #         tokenizer_or_name=mutate_tokenizer,
    #         model_type=MUTATE_MODEL_TYPE,
    #         num_new_prompts=N_NEW_PROMPTS
    #     )
    #     current_prompt_file = next_prompt_file
    
    print(f"\n[Generation {gen + 1}] Complete!")

print("\n" + "=" * 60)
print("ALL GENERATIONS COMPLETE!")
print("=" * 60)
print(f"Final best prompts: {best_prompt_output_path}")