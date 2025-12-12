#!/usr/bin/env python3
"""
Convert SFT dataset to OpenAI JSONL fine-tuning format.

The script converts a JSON dataset with instruction/input/output format
to OpenAI's fine-tuning format with messages array containing system, user, and assistant messages.
"""

import json
import argparse
import sys
from pathlib import Path


def convert_to_openai_format(input_file: str, output_file: str):
    """
    Convert dataset from SFT format to OpenAI JSONL format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
    """
    # Read input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        sys.exit(1)
    
    if not isinstance(data, list):
        print(f"Error: Expected a JSON array, got {type(data)}")
        sys.exit(1)
    
    # Convert each entry
    converted_data = []
    skipped = 0
    
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"Warning: Skipping entry {idx} - not a dictionary")
            skipped += 1
            continue
        
        # Extract fields
        instruction = entry.get('instruction', '')
        input_content = entry.get('input', '')
        output = entry.get('output', '')
        
        # Validate required fields
        if not instruction or not input_content or not output:
            print(f"Warning: Skipping entry {idx} - missing required fields")
            skipped += 1
            continue
        
        # Create OpenAI format
        openai_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": instruction.strip()
                },
                {
                    "role": "user",
                    "content": input_content.strip()
                },
                {
                    "role": "assistant",
                    "content": output.strip()
                }
            ]
        }
        
        converted_data.append(openai_entry)
    
    # Write to JSONL file (one JSON object per line)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in converted_data:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + '\n')
    except IOError as e:
        print(f"Error: Could not write to '{output_file}': {e}")
        sys.exit(1)
    
    print(f"Successfully converted {len(converted_data)} entries")
    if skipped > 0:
        print(f"Warning: Skipped {skipped} invalid entries")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SFT dataset to OpenAI JSONL fine-tuning format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: input_file with .jsonl extension)"
    )
    
    args = parser.parse_args()
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = str(input_path.with_suffix('.jsonl'))
    
    # Convert
    convert_to_openai_format(args.input_file, output_file)


if __name__ == "__main__":
    main()

