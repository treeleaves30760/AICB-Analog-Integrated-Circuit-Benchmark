# AICB: Analog Integrated Circuit Benchmark

AICB (Analog Integrated Circuit Benchmark) is a specialized benchmark designed to evaluate the performance of large language models (LLMs) on tasks related to analog integrated circuit design, analysis, and understanding.

## Overview

Analog integrated circuit design requires deep domain knowledge across multiple disciplines including semiconductor physics, circuit theory, and electronic systems. This benchmark tests an LLM's capability to reason about analog circuit concepts, solve circuit problems, and demonstrate understanding of fundamental analog IC principles.

## Dataset Structure

The benchmark dataset consists of multiple-choice questions covering various domains of analog integrated circuit design. Each question has four options (A, B, C, D) with one correct answer.

The dataset is provided in Parquet format with the following columns:

- `Question Number`: Unique identifier for each question
- `Question`: The question text
- `Option A`: First option
- `Option B`: Second option
- `Option C`: Third option
- `Option D`: Fourth option
- `Correct Answer`: The correct answer (A, B, C, or D)
- `Domain`: The specific analog IC domain the question belongs to

## Domains Covered

The benchmark covers several key domains in analog IC design:

- Amplifier Design (operational amplifiers, differential amplifiers)
- Noise Analysis
- Feedback Theory
- Frequency Response
- Stability and Compensation
- Biasing Techniques
- Bandgap References
- Analog-to-Digital and Digital-to-Analog Converters
- Oscillator Design
- Power Management Circuits
- Process, Voltage, and Temperature (PVT) Considerations

## Usage

### Install

```bash
pip install -r requirements.txt
```

Run the evaluator with the following command:

```bash
python llm_evaluator.py --provider [openai|anthropic|ollama] --model [model_name] [options]
```

### Arguments

- `--provider`: LLM provider (required, one of "openai", "anthropic", "ollama")
- `--model`: Model name (required, e.g., "gpt-4o", "claude-3-opus", "llama3")
- `--api-key`: API key for OpenAI or Anthropic (optional, can use environment variables)
- `--ollama-host`: Ollama host address (default: "<http://localhost:11434>")
- `--input-file`: Path to the Parquet file with test questions (default: "./data/test-000000-000001.parquet")
- `--output-dir`: Directory to save results (default: "./results")

### Environment Variables

You can set the following environment variables instead of passing API keys as arguments:

- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic

## Example Commands

### Evaluate with OpenAI GPT-4o

```bash
python llm_evaluator.py --provider openai --model gpt-4o
```

### Evaluate with Anthropic Claude

```bash
python llm_evaluator.py --provider anthropic --model claude-3-opus-20240229
```

### Evaluate with Ollama (local model)

```bash
python llm_evaluator.py --provider ollama --model llama3
```

## Input Data Format

The input Parquet file should have the following columns:

- Question Number: Unique identifier for each question
- Question: The question text
- Option A: First multiple-choice option
- Option B: Second multiple-choice option
- Option C: Third multiple-choice option
- Option D: Fourth multiple-choice option
- Correct Answer: The correct option (A, B, C, or D)
- Domain: The knowledge domain of the question

## Output Format

The evaluator generates a JSON file with the following structure:

```json
{
  "model": "model-name",
  "provider": "provider-name",
  "total_questions": 100,
  "correct_answers": 75,
  "overall_accuracy": 0.75,
  "domain_metrics": {
    "domain1": {
      "accuracy": 0.8,
      "correct": 20,
      "total": 25
    },
    "domain2": {
      "accuracy": 0.7,
      "correct": 35,
      "total": 50
    }
  },
  "detailed_results": [
    {
      "question_number": 1,
      "question": "Question text",
      "options": {
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      },
      "correct_answer": "A",
      "model_answer": "A",
      "raw_response": "The model's raw response",
      "is_correct": true,
      "domain": "domain1"
    }
  ]
}
```

## Notes

- The evaluator uses temperature=0 to get deterministic responses for evaluation
- The prompt is designed to elicit only the letter of the selected option
- The parser can handle various response formats from the models

## Citation

If you use AICB in your research, please cite:

```latex
@misc{aicb2025,
  author = {[Po-Hsiang, Hsu]},
  title = {AICB: Analog Integrated Circuit Benchmark},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{<https://github.com/treeleaves30760/AICB-Analog-Integrated-Circuit-Benchmark>}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions to improve the benchmark! Please see our CONTRIBUTING.md file for details on how to contribute.
