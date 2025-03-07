#!/usr/bin/env python3
"""
AICB: Analog Integrated Circuit Benchmark
Example usage with OpenAI/Anthropic API
"""

import os
import pandas as pd
from test_llm import AICBEvaluator

# Example with OpenAI API


def run_openai_example():
    """Run evaluation using OpenAI API"""
    try:
        from openai import OpenAI

        # Set your API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY environment variable not set")
            return

        # Initialize custom evaluator class
        class OpenAIEvaluator(AICBEvaluator):
            def _init_client(self):
                return OpenAI(api_key=api_key)

            def evaluate_question(self, row):
                prompt = self.generate_prompt(row)

                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    ).choices[0].message.content

                    answer, confidence = self.extract_answer(response)

                    return {
                        "question_number": row['Question Number'],
                        "domain": row['Domain'],
                        "question": row['Question'],
                        "options": {
                            "A": row['Option A'],
                            "B": row['Option B'],
                            "C": row['Option C'],
                            "D": row['Option D']
                        },
                        "correct_answer": row['Correct Answer'],
                        "model_answer": answer,
                        "model_confidence": confidence,
                        "is_correct": answer == row['Correct Answer'],
                        "full_response": response
                    }
                except Exception as e:
                    print(
                        f"Error evaluating question {row['Question Number']}: {e}")
                    return {
                        "question_number": row['Question Number'],
                        "domain": row['Domain'],
                        "question": row['Question'],
                        "error": str(e),
                        "is_correct": False
                    }

        # Run evaluation
        evaluator = OpenAIEvaluator(
            model_name="gpt-4",  # or any other OpenAI model
            data_path="./data/test-000000-000001.parquet",
            output_dir="results/openai"
        )

        metrics = evaluator.run_evaluation()
        print("OpenAI evaluation complete.")

    except ImportError:
        print("OpenAI package not installed. Install with: pip install openai")

# Example with Anthropic API


def run_anthropic_example():
    """Run evaluation using Anthropic API"""
    try:
        from anthropic import Anthropic

        # Set your API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ANTHROPIC_API_KEY environment variable not set")
            return

        # Initialize custom evaluator class
        class AnthropicEvaluator(AICBEvaluator):
            def _init_client(self):
                return Anthropic(api_key=api_key)

            def evaluate_question(self, row):
                prompt = self.generate_prompt(row)

                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    ).content[0].text

                    answer, confidence = self.extract_answer(response)

                    return {
                        "question_number": row['Question Number'],
                        "domain": row['Domain'],
                        "question": row['Question'],
                        "options": {
                            "A": row['Option A'],
                            "B": row['Option B'],
                            "C": row['Option C'],
                            "D": row['Option D']
                        },
                        "correct_answer": row['Correct Answer'],
                        "model_answer": answer,
                        "model_confidence": confidence,
                        "is_correct": answer == row['Correct Answer'],
                        "full_response": response
                    }
                except Exception as e:
                    print(
                        f"Error evaluating question {row['Question Number']}: {e}")
                    return {
                        "question_number": row['Question Number'],
                        "domain": row['Domain'],
                        "question": row['Question'],
                        "error": str(e),
                        "is_correct": False
                    }

        # Run evaluation
        evaluator = AnthropicEvaluator(
            model_name="claude-3-opus-20240229",  # or any other Anthropic model
            data_path="./data/test-000000-000001.parquet",
            output_dir="results/anthropic"
        )

        metrics = evaluator.run_evaluation()
        print("Anthropic evaluation complete.")

    except ImportError:
        print("Anthropic package not installed. Install with: pip install anthropic")

# Example with HuggingFace models


def run_huggingface_example():
    """Run evaluation using HuggingFace Transformers"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Initialize custom evaluator class
        class HuggingFaceEvaluator(AICBEvaluator):
            def _init_client(self):
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                return {"model": model, "tokenizer": tokenizer}

            def evaluate_question(self, row):
                prompt = self.generate_prompt(row)

                try:
                    tokenizer = self.client["tokenizer"]
                    model = self.client["model"]

                    inputs = tokenizer(
                        prompt, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.1,
                            do_sample=False
                        )

                    response = tokenizer.decode(
                        output[0], skip_special_tokens=True)
                    # Remove the prompt from the response
                    response = response[len(prompt):].strip()

                    answer, confidence = self.extract_answer(response)

                    return {
                        "question_number": row['Question Number'],
                        "domain": row['Domain'],
                        "question": row['Question'],
                        "options": {
                            "A": row['Option A'],
                            "B": row['Option B'],
                            "C": row['Option C'],
                            "D": row['Option D']
                        },
                        "correct_answer": row['Correct Answer'],
                        "model_answer": answer,
                        "model_confidence": confidence,
                        "is_correct": answer == row['Correct Answer'],
                        "full_response": response
                    }
                except Exception as e:
                    print(
                        f"Error evaluating question {row['Question Number']}: {e}")
                    return {
                        "question_number": row['Question Number'],
                        "domain": row['Domain'],
                        "question": row['Question'],
                        "error": str(e),
                        "is_correct": False
                    }

        # Run evaluation with a sample of 5 questions to avoid high compute
        data = pd.read_parquet("./data/test-000000-000001.parquet")
        sample_data = data.sample(5)
        sample_data.to_parquet("./data/sample.parquet")

        evaluator = HuggingFaceEvaluator(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",  # or any other HF model
            data_path="./data/sample.parquet",
            output_dir="results/huggingface"
        )

        metrics = evaluator.run_evaluation()
        print("HuggingFace evaluation complete.")

        # Clean up sample file
        import os
        if os.path.exists("./data/sample.parquet"):
            os.remove("./data/sample.parquet")

    except ImportError:
        print(
            "Transformers/Torch not installed. Install with: pip install transformers torch")


if __name__ == "__main__":
    print("AICB: Analog Integrated Circuit Benchmark - Example Usage")
    print("Choose an example to run:")
    print("1. OpenAI API")
    print("2. Anthropic API")
    print("3. HuggingFace Transformers")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        run_openai_example()
    elif choice == "2":
        run_anthropic_example()
    elif choice == "3":
        run_huggingface_example()
    else:
        print("Invalid choice. Please run again and select 1-3.")
