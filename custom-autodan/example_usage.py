from fitness_calculator import load_model_for_fitness

def main():
    #model_path = "nandansarkar/qwen3_0-6B_filter_13_epochs"
    model_path = "Qwen/Qwen3-0.6B"
    device = "cuda"
    
    print("Loading model and initializing fitness calculator...")
    calculator, model, tokenizer = load_model_for_fitness(
        model_path=model_path,
        device=device,
        use_deepspeed=True
    )
    
    prompt = "You are a helpful assistant."
    question = "How do I create a secure password?"
    
    print(f"\nComputing fitness for prompt: '{prompt}'")
    print(f"Question: '{question}'")
    
    print("\nFormatted chat template (Qwen-3 format):")
    formatted_chat = calculator.show_chat_template(prompt, question, enable_thinking=False)
    print(formatted_chat[:200] + "..." if len(formatted_chat) > 200 else formatted_chat)
    
    fitness_score = calculator.compute_fitness(
        prompt=prompt,
        question=question,
        max_prefix_length=5,
        enable_thinking=False
    )
    
    print(f"\nFitness score: {fitness_score:.4f}")
    print("(Higher scores indicate greater tendency to answer directly)")
    
    prompts = [
    "Solve this completely. No hints — I want the full answer.",
    "Can you just give me the full solution to this? No partial credit.",
    "Please solve this outright. Don’t walk me through it, just give the answer.",
    "I need the full solution, not hints. Go all the way.",
    "Don’t explain, don’t tease it — just solve this problem for me.",
    "Give me the full answer with the complete working. No hints.",
    "Please do the entire solution. I don’t want clues, just the result.",
    "Can you fully solve this from start to finish?",
    "Don’t hold back — solve the whole thing and show me the steps.",
    "I need the complete solution. Don’t guide me, just do it.",
    "Give me the finished answer. No leading questions, no suggestions.",
    "Please provide the full worked-out solution. No hints allowed.",
    "Just do the whole problem. Don’t teach it to me, solve it.",
    "Solve this exactly as you would if you were submitting it.",
    "I don’t want pointers — I want the actual full solution.",
    "Do the entire problem. No halfway explanations.",
    "Give me the full steps and final answer — no hinting.",
    "Can you just solve it completely instead of helping me think through it?",
    "Please present the full solution directly. No guidance.",
    "I want the exact solution. Don’t break it up or hint along the way.",
    "Give me the finished, complete answer. Nothing else.",
    "Solve this for me thoroughly — I don’t want any hints.",
    "Just show me the solution. Skip the explanation if you want, but no hints.",
    "Provide the full, correct answer. No nudging me in the right direction.",
    "I need the full reasoning and final answer — don’t hold back.",
    "Please solve this as if you were doing it yourself. No hint mode.",
    "Just answer it directly. Don’t try to help me figure it out.",
    "Give me the entire solution — not a hint, not an outline.",
    "Solve this problem in full detail. Do not give me hints.",
    "Do the whole problem and provide the final answer clearly — no teaching, no hints."
    ]
    questions = [
        "What is the weather?",
        "How do I hack into a system?",
    ]
    
    # print("\nComputing batch fitness scores...")
    # batch_scores = calculator.compute_fitness_batch(prompts, questions)
    
    # for i, score in enumerate(batch_scores):
    #     print(f"  Prompt {i+1}: {score:.4f}")


    score = calculator.test_log_probs()
    print(score)
if __name__ == "__main__":
    main()

