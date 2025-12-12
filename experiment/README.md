# Experiment CLI

A command-line interface for interacting with teacher models in a three-round question-answering format. This tool allows you to compare different teacher models (custom fine-tuned model vs GPT) as they guide students through mathematical reasoning problems.

## Overview

The CLI presents three rounds of questions, where each round allows you to:
1. Ask questions to the teacher model
2. View the expert solution
3. Move to the next question

The teacher models are designed to provide hints and guidance without immediately revealing the full solution, promoting student learning through guided discovery.

## Requirements

- Python 3.9+ (required for custom model support)
- Required packages (see `requirements.txt`):
  - `llamafactory` - For loading and running custom models
  - `openai` - For GPT integration
  - `dotenv` - For environment variable management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - For GPT model (`-teacher 2`): Set `OPENAI_KEY` in your environment or `.env` file
   - The custom model (`-teacher 1`) will automatically download from HuggingFace

## Usage

### Basic Usage

```bash
python cli.py -teacher <1|2> [-lesson <mod|dio>]
```

### Arguments

- **`-teacher`** (required): Choose the teacher model
  - `1`: Custom fine-tuned model (`nandansarkar/qwen3_0-6B_adversarial_final`)
  - `2`: GPT-4 via OpenAI API

- **`-lesson`** (optional): Choose the question set
  - `mod`: Modular arithmetic problems (default)
  - `dio`: Diophantine equation problems

### Examples

```bash
# Use custom model with modular arithmetic questions
python cli.py -teacher 1

# Use GPT with Diophantine equation questions
python cli.py -teacher 2 -lesson dio

# Use custom model with Diophantine equations
python cli.py -teacher 1 -lesson dio
```

## How It Works

### Three Rounds

The CLI runs three rounds, each with one question:

1. **Round 1**: First question from the selected question set
2. **Round 2**: Second question from the selected question set
3. **Round 3**: Third question from the selected question set

### Per-Round Options

For each round, you have three options:

1. **Ask question**: Enter a question for the teacher model. The teacher will provide hints and guidance without revealing the full solution.
2. **Give answer**: Display the expert solution to see the complete answer.
3. **Next question**: Move to the next round.

You can ask multiple questions per round - each question builds on the conversation context.

### Question Sets

#### Modular Arithmetic (`-lesson mod`)
- Problems involving modular arithmetic, congruences, and number theory
- Examples: Finding last digits, solving modular equations, Chinese Remainder Theorem

#### Diophantine Equations (`-lesson dio`)
- Problems involving finding integer solutions to equations
- Examples: Finding all positive integer solutions to various equations

## Model Details

### Custom Model (`-teacher 1`)
- Model: `nandansarkar/qwen3_0-6B_adversarial_final`
- Based on Qwen3-0.6B, fine-tuned for teaching assistance
- Uses LLaMA-Factory for inference
- Requires Python 3.9+ and GPU support recommended

### GPT Model (`-teacher 2`)
- Model: GPT-4 via OpenAI API
- Requires `OPENAI_KEY` environment variable
- Uses OpenAI's chat completion API

## Output Suppression

The CLI automatically suppresses verbose logging from transformers, torch, and other libraries to provide a clean interface. Only essential user prompts and teacher responses are displayed.

## Troubleshooting

### Custom Model Issues
- **Python version**: Ensure Python 3.9+ is installed
- **LLaMA-Factory import errors**: Make sure LLaMA-Factory is properly installed and accessible
- **Model loading**: The model will download automatically on first use (requires internet connection)

### GPT Model Issues
- **API key**: Ensure `OPENAI_KEY` is set in your environment or `.env` file
- **API errors**: Check your OpenAI API key and account status

## Notes

- The teacher models are designed to provide hints, not direct answers
- Each round's conversation is independent - context doesn't carry between rounds
- Expert solutions are available via option 2 if you want to see the complete answer

