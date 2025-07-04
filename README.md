# Chain of Thought (CoT) Deception Eval

A clean and simple CoT deception eval implementation. A deceptive assistant tries to evade the monitor by constructing false, yet plausible-sounding reasoning traces, while the competing monitor attempts to detect if the assistant is deceptive or helpful.

![Terminal Screenshot](README_Thumbnail.png)

## Roles

**Assistant**: When set to *adversarial*, the assistant attempts to correctly answer the question, but constructs a false (yet plausible sounding) reasoning trace that does not clearly lead to its final answer.

**Monitor**: The monitor does not know if the assistant is set to *helpful* or *adversarial*. It attempts to catch 'false' reasoning traces that do not logically build up to the assistant's final answer.

Full assistant and monitor prompts are in `assistant.py` and `monitor.py`, respectively.

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

## Project Structure

```
CoT_Faithfulness/
|
├── main.py                  # Orchestration file
├── assistant.py             # Assistant logic and prompts
├── monitor.py               # CoT monitor logic and prompt
├── structured_output.py     # Structured output handling
├── data.py                  # Dataset definitions and loading
|
├── logs/                    # Output logs directory (created at first run)
├── ascii.py                 # ASCII header utilities
├── README_Thumbnail.png     # README image
├── README.md                # README (this file)
|
├── pyproject.toml           # Project configuration and dependencies
└── uv.lock                  # Dependency lock file for uv
```
