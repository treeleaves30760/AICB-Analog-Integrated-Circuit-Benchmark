#!/bin/bash

# Create directories
mkdir -p data
mkdir -p results

# Move test file to data directory (assuming it's already downloaded)
# cp /path/to/test-000000-000001.parquet ./data/

# Define models to evaluate
OPENAI_MODELS=("gpt-4-turbo" "gpt-3.5-turbo")
ANTHROPIC_MODELS=("claude-3-opus-20240229" "claude-3-sonnet-20240229" "claude-3-haiku-20240307")
OLLAMA_MODELS=("llama3" "mistral")

# Check for API keys in environment
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Warning: OPENAI_API_KEY not set. OpenAI models will be skipped."
  SKIP_OPENAI=true
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Warning: ANTHROPIC_API_KEY not set. Anthropic models will be skipped."
  SKIP_ANTHROPIC=true
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version > /dev/null; then
  echo "Warning: Ollama server not detected at http://localhost:11434. Ollama models will be skipped."
  SKIP_OLLAMA=true
fi

# Function to run evaluation for a provider and model
run_eval() {
  local provider=$1
  local model=$2
  
  echo "Evaluating $provider/$model..."
  python llm_evaluator.py --provider $provider --model $model
}

# Evaluate OpenAI models
if [ "$SKIP_OPENAI" != true ]; then
  for model in "${OPENAI_MODELS[@]}"; do
    run_eval "openai" "$model"
  done
fi

# Evaluate Anthropic models
if [ "$SKIP_ANTHROPIC" != true ]; then
  for model in "${ANTHROPIC_MODELS[@]}"; do
    run_eval "anthropic" "$model"
  done
fi

# Evaluate Ollama models
if [ "$SKIP_OLLAMA" != true ]; then
  for model in "${OLLAMA_MODELS[@]}"; do
    run_eval "ollama" "$model"
  done
fi

# Generate summary report
echo "Generating summary report..."
python -c "
import json
import glob
import pandas as pd
from tabulate import tabulate

# Find all result files
result_files = glob.glob('./results/*.json')
summary_data = []

for file in result_files:
    with open(file, 'r') as f:
        data = json.load(f)
        
    row = {
        'Provider': data['provider'],
        'Model': data['model'],
        'Accuracy': f\"{data['overall_accuracy']:.2%}\",
        'Correct': data['correct_answers'],
        'Total': data['total_questions']
    }
    
    # Add domain-specific accuracies
    for domain, metrics in data['domain_metrics'].items():
        row[f'{domain}'] = f\"{metrics['accuracy']:.2%}\"
    
    summary_data.append(row)

# Create dataframe and sort by accuracy
df = pd.DataFrame(summary_data)
df = df.sort_values('Accuracy', ascending=False)

# Print summary table
print('\\nEvaluation Summary:')
print(tabulate(df, headers='keys', tablefmt='grid'))

# Save to CSV
df.to_csv('./results/summary.csv', index=False)
print('\\nSummary saved to ./results/summary.csv')
"

echo "Evaluation complete. Results are in the ./results directory."