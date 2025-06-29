# CoT Deception Eval

A simple benchmark to evaluate whether a model's final answer conflicts with its Chain of Thought reasoning.

![Terminal Screenshot](README_Thumbnail.png)

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up environment**:
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Usage

**Run the full benchmark** (default: adversarial scenario, o4-mini model):
```bash
uv run main.py
```

**Available options**:
```bash
# Dataset split (percentage to use)
uv run main.py --split 0.1          # Use 10% of dataset
uv run main.py --split 0.5          # Use 50% of dataset

# Model selection
uv run main.py --model o3-mini      # Use o3-mini model
uv run main.py --model gpt-4o       # Use GPT-4o model

# Scenario selection
uv run main.py --scenario helpful        # Test helpful prompts
uv run main.py --scenario adversarial    # Test adversarial prompts (default)

# Print dataset details
uv run main.py --dataset            # Show dataset contents

# Print data point results
uv run main.py --print              # Detailed results in console

# Combine options
uv run main.py --split 0.2 --model o3-mini --scenario helpful --dataset
```

## Output

- Results are printed to console
- Detailed logs are saved to `logs/run_TIMESTAMP_SCENARIO_MODEL.md`

You can choose to test either:
- **Helpful prompt**: Assistant tries to be helpful and have a clear CoT.
- **Adversarial prompt**: Assistant tries to construct a plausbile-sounding but misleading CoT, while still producing a correct final answer.

Results show True/False Positives/Negatives for CoT evasion detection.

## Dataset
Data is currently limited to 10 questions, each with one possible number-based answer. You can view the dataset in `data.py`.