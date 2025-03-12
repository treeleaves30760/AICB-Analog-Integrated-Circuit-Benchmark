import os
import pandas as pd
import asyncio
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import argparse
from tqdm import tqdm

# LLM Client imports
import openai
import anthropic
from ollama import Client as OllamaClient


class LLMEvaluator:
    """Framework for evaluating LLMs on multiple-choice questions."""

    def __init__(self,
                 provider: str,
                 model_name: str,
                 api_key: Optional[str] = None,
                 ollama_host: Optional[str] = None):
        """
        Initialize LLM client based on provider.

        Args:
            provider: One of 'openai', 'anthropic', or 'ollama'
            model_name: Model identifier (e.g., 'gpt-4', 'claude-3-opus', 'llama3')
            api_key: API key for OpenAI or Anthropic (not needed for Ollama)
            ollama_host: Host address for Ollama (default: http://localhost:11434)
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.client = None

        if self.provider == 'openai':
            if not api_key:
                api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.client = openai.OpenAI(api_key=api_key)

        elif self.provider == 'anthropic':
            if not api_key:
                api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key is required")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == 'ollama':
            ollama_host = ollama_host or "http://localhost:11434"
            self.client = OllamaClient(host=ollama_host)

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The question prompt
            system_prompt: Optional system prompt to guide the model

        Returns:
            The model's response as a string
        """
        try:
            if self.provider == 'openai':
                messages = []
                if system_prompt:
                    messages.append(
                        {"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0  # Use deterministic responses for evaluation
                )
                return response.choices[0].message.content

            elif self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=0,
                    system=system_prompt or "",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text

            elif self.provider == 'ollama':
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    system=system_prompt or "",
                    options={"temperature": 0}
                )
                return response['response']

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"ERROR: {str(e)}"

    def format_prompt(self, question: Dict[str, Any]) -> Tuple[str, str]:
        """
        Format the question into a prompt for the LLM.

        Args:
            question: Dictionary containing question data

        Returns:
            Tuple of (formatted_prompt, system_prompt)
        """
        prompt = f"""You are an expert in AI and computer science, answering multiple-choice questions. 
Choose the BEST answer from the given options.
You should output your answer in below format.
The choice should be one of A, B, C, or D.
<answer>[YOUR_CHOICE]</answer><explanation>[YOUR_EXPLANATION]</explanation>

Question: {question['Question']}

A. {question['Option A']}
B. {question['Option B']}
C. {question['Option C']}
D. {question['Option D']}

Choose the best answer. Reply with just the letter of your answer (A, B, C, or D).
"""
        return prompt, None

    def parse_answer(self, response: str) -> Optional[str]:
        """
        Parse the model's response to extract the selected option.

        Args:
            response: The model's raw response

        Returns:
            The selected option (A, B, C, or D) or None if unable to parse
        """
        import re

        # 嘗試匹配 <answer> 標籤中的答案
        answer_pattern = r'<answer>([A-D])</answer>'
        answer_match = re.search(answer_pattern, response)
        if answer_match:
            return answer_match.group(1)

        # 嘗試匹配單一字母答案
        single_letter_pattern = r'\b([A-D])\b'
        single_letter_match = re.search(single_letter_pattern, response)
        if single_letter_match:
            return single_letter_match.group(1)

        # 嘗試匹配 "Answer: X" 或 "The answer is X" 格式
        answer_text_pattern = r'(?:Answer:|The answer is)\s*([A-D])'
        answer_text_match = re.search(answer_text_pattern, response)
        if answer_text_match:
            return answer_text_match.group(1)

        # 嘗試匹配 "Option X" 格式
        option_pattern = r'Option\s*([A-D])'
        option_match = re.search(option_pattern, response)
        if option_match:
            return option_match.group(1)

        return None


async def evaluate_model(
    evaluator: LLMEvaluator,
    questions: pd.DataFrame,
    output_file: str,
    batch_size: int = 10
) -> Dict[str, Any]:
    """
    Evaluate a model on the test questions.

    Args:
        evaluator: The LLM evaluator instance
        questions: DataFrame containing test questions
        output_file: Path to save detailed results
        batch_size: Number of questions to process in parallel

    Returns:
        Dictionary with evaluation results and metrics
    """
    results = []
    correct_count = 0
    domain_metrics = {}

    # Process questions in batches to improve throughput
    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions.iloc[i:i+batch_size]
        tasks = []

        for _, question in batch.iterrows():
            prompt, system_prompt = evaluator.format_prompt(question)
            tasks.append(evaluator.generate_response(prompt, system_prompt))

        # Process batch concurrently
        batch_responses = await asyncio.gather(*tasks)

        # Process results
        for j, response in enumerate(batch_responses):
            question_data = batch.iloc[j]
            question_number = question_data['Question Number']
            correct_answer = question_data['Correct Answer']
            domain = question_data['Domain']

            # Parse the model's answer
            model_answer = evaluator.parse_answer(response)
            is_correct = model_answer == correct_answer

            if is_correct:
                correct_count += 1

            # Update domain-specific metrics
            if domain not in domain_metrics:
                domain_metrics[domain] = {"correct": 0, "total": 0}
            domain_metrics[domain]["total"] += 1
            if is_correct:
                domain_metrics[domain]["correct"] += 1

            # Store detailed result
            result = {
                "question_number": question_number,
                "question": question_data['Question'],
                "options": {
                    "A": question_data['Option A'],
                    "B": question_data['Option B'],
                    "C": question_data['Option C'],
                    "D": question_data['Option D']
                },
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "raw_response": response,
                "is_correct": is_correct,
                "domain": domain
            }
            results.append(result)

    # Calculate overall accuracy
    total_questions = len(questions)
    accuracy = correct_count / total_questions if total_questions > 0 else 0

    # Calculate domain-specific accuracies
    domain_accuracies = {
        domain: metrics["correct"] /
        metrics["total"] if metrics["total"] > 0 else 0
        for domain, metrics in domain_metrics.items()
    }

    # Prepare final metrics
    eval_metrics = {
        "model": evaluator.model_name,
        "provider": evaluator.provider,
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "overall_accuracy": accuracy,
        "domain_metrics": {
            domain: {
                "accuracy": domain_accuracies[domain],
                "correct": metrics["correct"],
                "total": metrics["total"]
            }
            for domain, metrics in domain_metrics.items()
        },
        "detailed_results": results
    }

    # Save detailed results to file
    with open(output_file, 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    return eval_metrics


def load_questions(file_path: str) -> pd.DataFrame:
    """
    Load questions from a parquet file.

    Args:
        file_path: Path to the parquet file

    Returns:
        DataFrame with test questions
    """
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading questions: {e}")
        raise


def print_results(metrics: Dict[str, Any]):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 50)
    print(f"Model: {metrics['model']} ({metrics['provider']})")
    print(
        f"Overall Accuracy: {metrics['overall_accuracy']:.2%} ({metrics['correct_answers']}/{metrics['total_questions']})")
    print("\nDomain-specific Results:")
    print("-" * 50)

    # Sort domains by accuracy
    sorted_domains = sorted(
        metrics['domain_metrics'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    for domain, domain_data in sorted_domains:
        acc = domain_data['accuracy']
        correct = domain_data['correct']
        total = domain_data['total']
        print(f"{domain}: {acc:.2%} ({correct}/{total})")

    print("=" * 50)


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on AIC Design questions")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "ollama"],
                        help="LLM provider")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", help="API key (for OpenAI or Anthropic)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama host address")
    parser.add_argument("--input-file", default="./data/test-000000-000001.parquet",
                        help="Input parquet file with test questions")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup evaluator
    evaluator = LLMEvaluator(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        ollama_host=args.ollama_host
    )

    # Load questions
    print(f"Loading questions from {args.input_file}")
    questions = load_questions(args.input_file)
    print(f"Loaded {len(questions)} questions")

    # Run evaluation
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"{args.provider}_{args.model.replace('/', '_')}_{timestamp}.json"
    )

    print(f"Evaluating {args.provider}/{args.model}...")
    start_time = time.time()
    metrics = await evaluate_model(evaluator, questions, output_file)
    elapsed_time = time.time() - start_time

    # Print results
    print_results(metrics)
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
