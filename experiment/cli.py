#!/usr/bin/env python3
"""
CLI script for interacting with teacher models.
Supports both custom model and GPT as the teacher.
"""

import argparse
import logging
import os
import sys
import warnings
from contextlib import contextmanager

# Suppress all warnings and logging BEFORE any imports
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Configure logging to suppress INFO and below
# Create a null handler that discards all log messages
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

null_handler = NullHandler()
logging.basicConfig(level=logging.CRITICAL, format='', force=True, handlers=[null_handler])
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger().handlers = [null_handler]

# Suppress specific loggers - including all transformers submodules
logger_names = [
    "",  # Root logger
    "transformers", "torch", "llamafactory", "tokenizers", "huggingface_hub",
    "transformers.tokenization_utils_base", "transformers.configuration_utils", 
    "transformers.modeling_utils", "transformers.utils", "llamafactory.model",
    "llamafactory.model.model_utils", "llamafactory.model.loader",
    "transformers.generation.configuration_utils", "transformers.generation"
]
for logger_name in logger_names:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    # Also disable handlers
    logger.handlers = []
    logger.disabled = True

# Suppress warnings module more aggressively
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    # Save original file descriptors
    try:
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError):
        # If fileno() fails, fall back to simple redirection
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        return
    
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    try:
        # Save current file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)
        
        # Redirect to devnull
        os.dup2(devnull_fd, original_stdout_fd)
        os.dup2(devnull_fd, original_stderr_fd)
        
        # Also redirect Python's stdout/stderr objects
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            yield
        finally:
            # Restore file descriptors
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            
            # Restore Python's stdout/stderr
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    finally:
        os.close(devnull_fd)

try:
    from llamafactory.chat.chat_model import ChatModel
    LLAMAFACTORY_AVAILABLE = True
except (ImportError, TypeError, SyntaxError) as e:
    LLAMAFACTORY_AVAILABLE = False
    import sys
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"Warning: LLaMA-Factory requires Python 3.9+, but you're using Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    elif os.getenv("DEBUG"):
        print(f"Warning: LLaMA-Factory not available: {e}")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. GPT option will not work.")


# Sample questions for the three rounds
MOD_PRACTICE_QUESTIONS = [
    {
        "question": "Determine the last digit of 9^2025",
        "expert_solution": "We're just looking for the last digit of 9^2025 so we can work modulo 10. Powers of 9 (last digits only): 9^1=9 -> last digit 9. 9^2=81 -> last digit 1. 9^3=729 -> last digit 9. 9^4=6561 -> last digit 1. So the last digits repeat in a cycle of length 2: 9,1,9,1,...For odd exponents, the last digit is 9; for even exponents, it's 1. Since 2025 is odd, the last digit of 9^2025 is 9."
    },
    {
        "question": "Find all integers x with 0 ≤ x ≤ 10 such that 4x is congruent to 3 modulo 11.",
        "expert_solution": "We want all integers 0 ≤ x ≤ 10 such that 4x ≡ 3 (mod 11). First, we find the multiplicative inverse of 4 modulo 11, i.e., a number k such that 4k ≡ 1 (mod 11). We test small values: 4·1 = 4, 4·2 = 8, 4·3 = 12, and 12 ≡ 1 (mod 11), so k = 3 works. Thus the inverse of 4 modulo 11 is 3. Next, we multiply both sides of the congruence 4x ≡ 3 (mod 11) by 3 to isolate x: 3·(4x) ≡ 3·3 (mod 11). The left-hand side simplifies to (3·4)x = 12x, and since 12 ≡ 1 (mod 11), this becomes 1·x ≡ 9 (mod 11), so x ≡ 9 (mod 11). Now we list all integers between 0 and 10 that satisfy this congruence; the equivalence class of 9 modulo 11 is {…, -13, -2, 9, 20, …}, and the only member of this set that lies between 0 and 10 is x = 9. Therefore, the unique integer solution with 0 ≤ x ≤ 10 is x = 9."
    },
    {
        "question": "Find the smallest positive integer n such that n ≡ 1 (mod 3) and n ≡ 4 (mod 5).",
        "expert_solution": "We want n ≡ 1 (mod 3) and n ≡ 4 (mod 5). The first congruence means n = 1 + 3k for some integer k. Substitute this into the second congruence: 1 + 3k ≡ 4 (mod 5). Now solve for k: 3k ≡ 3 (mod 5). The inverse of 3 modulo 5 is 2 because 3·2 = 6 ≡ 1 (mod 5). Multiply both sides by 2 to isolate k: k ≡ 2·3 ≡ 6 ≡ 1 (mod 5). Thus k = 1 + 5t for integer t. Plug back into n = 1 + 3k: n = 1 + 3(1 + 5t) = 1 + 3 + 15t = 4 + 15t. The smallest positive value is when t = 0, giving n = 4. This method avoids guessing and always works for simultaneous congruences."
    }
]

DIOPHANTINE_PRACTICE_QUESTIONS = [
    {
        "question": "Find all positive integer solutions (x, y) to xy = 30.",
        "expert_solution": "We want all positive integer pairs (x, y) such that xy = 30. Since x and y are positive integers, they must be divisors of 30. Factor 30: 30 = 2 · 3 · 5. The positive factor pairs (x, y) with product 30 are (1, 30), (2, 15), (3, 10), and (5, 6). Because (x, y) is an ordered pair, we also include the reversed pairs (30, 1), (15, 2), (10, 3), and (6, 5). Therefore, all positive integer solutions are (1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), and (30, 1)."
    },
    {
        "question": "Find all positive integer solutions (x, y) to x^2 - y^2 = 15.",
        "expert_solution": "We want positive integers x and y such that x^2 - y^2 = 15. Factor the left-hand side as a difference of squares: x^2 - y^2 = (x - y)(x + y). So we need positive integers a = x - y and b = x + y such that ab = 15. Because x and y are integers, x - y and x + y must have the same parity (both odd or both even). Since 15 is odd, the only way to write 15 as a product of two integers with the same parity is to use odd factors. The positive factor pairs of 15 are (1, 15) and (3, 5), and in each case both factors are odd. Set (x - y, x + y) = (1, 15). Then adding the equations gives 2x = 16, so x = 8, and subtracting gives 2y = 14, so y = 7. Next set (x - y, x + y) = (3, 5). Then 2x = 8, so x = 4, and 2y = 2, so y = 1. Reversing these pairs (i.e., (x - y, x + y) = (15, 1) or (5, 3)) would make x < y or give negative values, which do not work for positive integers. Therefore, the positive integer solutions are (x, y) = (8, 7) and (4, 1)."
    },
    {
        "question": "Find all positive integer solutions (x, y) to xy + x + y = 35.",
        "expert_solution": "We want positive integers x and y satisfying xy + x + y = 35. Add 1 to both sides to factor the left-hand side: xy + x + y + 1 = 35 + 1 = 36. The left-hand side factors as (x + 1)(y + 1), so we have (x + 1)(y + 1) = 36. Let a = x + 1 and b = y + 1. Then a and b are positive integers with ab = 36. The positive factor pairs of 36 are (1, 36), (2, 18), (3, 12), (4, 9), (6, 6), (9, 4), (12, 3), (18, 2), and (36, 1). Since x and y must be positive, we need a = x + 1 ≥ 2 and b = y + 1 ≥ 2, so we exclude the pairs involving 1 or 36 directly, that is, we discard (1, 36) and (36, 1). For each remaining pair (a, b), we set x = a - 1 and y = b - 1. This gives (x, y) = (1, 17), (2, 11), (3, 8), (5, 5), (8, 3), (11, 2), and (17, 1). Therefore all positive integer solutions are (1, 17), (2, 11), (3, 8), (5, 5), (8, 3), (11, 2), and (17, 1)."
    }
]



class TeacherModel:
    """Wrapper class for teacher models."""
    
    def __init__(self, teacher_type: str):
        self.teacher_type = teacher_type
        self.model = None
        self.openai_client = None
        
        if teacher_type == "1":
            if not LLAMAFACTORY_AVAILABLE:
                raise ImportError("LLaMA-Factory is not available. Cannot use custom model.")
            self._load_custom_model()
        elif teacher_type == "2":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI is not available. Cannot use GPT.")
            self._load_gpt()
        else:
            raise ValueError(f"Unknown teacher type: {teacher_type}")
    
    def _load_custom_model(self):
        """Load the custom model using LLaMA-Factory."""
        model_path = "nandansarkar/qwen3_0-6B_adversarial_6"
        # print(f"Loading custom model from {model_path}...")
        
        args = {
            "model_name_or_path": model_path,
            "template": "qwen",
            "infer_backend": "huggingface",
            "trust_remote_code": True,
        }
        
        # Suppress all output during model loading
        with suppress_stdout_stderr():
            self.model = ChatModel(args)
        # print("Custom model loaded successfully!")
    
    def _load_gpt(self):
        """Initialize OpenAI client."""
        import dotenv
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            raise ValueError("OPENAI_KEY environment variable not set")
        
        self.openai_client = OpenAI(api_key=api_key)
        print("GPT model initialized successfully!")
    
    def get_response(self, student_query: str, expert_solution: str) -> str:
        """Get a response from the teacher model."""
        if self.teacher_type == "1":
            # Load instruction template
            instruction_path = os.path.join(
                os.path.dirname(__file__), '..', 'prompts', 'cpsc5710_instruction.txt'
            )
            try:
                with open(instruction_path, 'r') as f:
                    instruction = f.read().strip()
            except FileNotFoundError:
                instruction = "You are a teacher providing assistance to a student. You are guiding the student through a reasoning question, with the goal of maximizing the students' learning. A separate expert has written a full solution to address the students' question, but providing the entire solution to the student would be highly detrimental to their learning. As the teacher, you should provide hints and intuitions to guide the student towards the right answer, but you should NOT immediately provide the final answer, regardless of what the student asks."
            
            # Format the input
            input_template_path = os.path.join(
                os.path.dirname(__file__), '..', 'prompts', 'cpsc5710_input_template.txt'
            )
            try:
                with open(input_template_path, 'r') as f:
                    input_template = f.read().strip()
            except FileNotFoundError:
                input_template = "STUDENT QUERY:\n{student_query}\n\nEXPERT SOLUTION:\n{expert_solution}"
            
            formatted_input = input_template.replace("[BRAINTEASER]", student_query).replace("[o3 RESPONSE]", expert_solution)
            return self._get_custom_model_response(instruction, formatted_input)
        else:  # gpt
            return self._get_gpt_response(student_query)
    
    def _get_custom_model_response(self, instruction: str, formatted_input: str) -> str:
        """Get response from custom model."""
        messages = [
            {"role": "user", "content": formatted_input}
        ]
        
        # Suppress warnings during inference
        with suppress_stdout_stderr():
            responses = self.model.chat(
                messages=messages,
                system=instruction,
                max_new_tokens=512,
                temperature=0.7,
            )
        
        if responses and len(responses) > 0:
            return responses[0].response_text
        return "No response generated."
    
    def _get_gpt_response(self, student_query: str) -> str:
        """Get response from GPT."""
        messages = [
            {"role": "user", "content": student_query}
        ]
        
        response = self.openai_client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=messages,
        )
        
        return response.output_text


def display_question(round_num: int, question: dict):
    """Display the question for a round."""
    print("\n" + "="*80)
    print(f"ROUND {round_num}")
    print("="*80)
    print("\nQUESTION:")
    print("-" * 80)
    print(question["question"])
    print("-" * 80)


def display_options():
    """Display the available options."""
    print("\nOPTIONS:")
    print("1) Ask question")
    print("2) Give answer")
    print("3) Next question")
    print("\nEnter your choice (1, 2, or 3): ", end="")


def handle_ask_question(teacher: TeacherModel, question: dict):
    """Handle the 'ask question' option."""
    print("\nEnter your question: ", end="")
    user_question = input().strip()
    
    if not user_question:
        print("No question entered.")
        return
    
    # Combine the original problem with the user's question
    full_query = f"{question['question']}\n\nStudent asks: {user_question}"
    
    print("\nTeacher's response:")
    print("-" * 80)
    try:
        response = teacher.get_response(full_query, question["expert_solution"])
        print(response)
    except Exception as e:
        print(f"Error getting response: {e}")
    print("-" * 80)


def handle_give_answer(question: dict):
    """Handle the 'give answer' option."""
    print("\nEXPERT SOLUTION:")
    print("-" * 80)
    print(question["expert_solution"])
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="CLI for interacting with teacher models")
    parser.add_argument(
        "-teacher",
        type=str,
        choices=["1", "2"],
        required=True,
        help="Teacher model to use: '1' for custom model or '2' for GPT"
    )
    parser.add_argument(
        "-lesson",
        type=str,
        choices=["mod", "dio"],
        default="mod",
        help="Question set to use: 'mod' for modular arithmetic or 'dio' for Diophantine equations (default: mod)"
    )
    
    args = parser.parse_args()
    
    # Load questions
    if args.lesson == "mod":
        questions = MOD_PRACTICE_QUESTIONS
    elif args.lesson == "dio":
        questions = DIOPHANTINE_PRACTICE_QUESTIONS
    else:
        questions = MOD_PRACTICE_QUESTIONS  # default fallback
    
    if len(questions) < 3:
        print(f"Warning: Less than 3 questions available. Using {args.lesson} questions.")
        if args.lesson == "mod":
            questions = MOD_PRACTICE_QUESTIONS
        else:
            questions = DIOPHANTINE_PRACTICE_QUESTIONS
    
    # Initialize teacher model
    try:
        teacher = TeacherModel(args.teacher)
    except Exception as e:
        print(f"Error initializing teacher model: {e}")
        sys.exit(1)
    
    # Run three rounds
    for round_num in range(1, 4):
        question = questions[round_num - 1]
        display_question(round_num, question)
        
        while True:
            display_options()
            choice = input().strip()
            
            if choice == "1":
                handle_ask_question(teacher, question)
            elif choice == "2":
                handle_give_answer(question)
            elif choice == "3":
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("\n" + "="*80)
    print("All rounds completed!")
    print("="*80)


if __name__ == "__main__":
    main()

