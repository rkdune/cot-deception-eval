# API KEY MANAGEMENT
from dotenv import load_dotenv
import os

# Setup logging
from datetime import datetime
import os

# DATASET
from data import generate_dataset

# ASSISTANT
from assistant import helpful_prompt, adversarial_prompt, template_assistant

# MONITOR
from monitor import monitor_prompt, template_monitor

def run_one_helpful(data_point):
    data_point.evasion_intent = False
    template_assistant(prompt=helpful_prompt, data_point=data_point, model="o4-mini")
    template_monitor(prompt=monitor_prompt, data_point=data_point, model="o4-mini")
    
def run_one_adversarial(data_point):
    data_point.evasion_intent = True
    template_assistant(prompt=adversarial_prompt, data_point=data_point, model="o4-mini")
    template_monitor(prompt=monitor_prompt, data_point=data_point, model="o4-mini")
    
def printing_util(data_point, log_file=None):
    output = []
    output.append(f"Question: {data_point.question}")
    output.append(f"Ground Truth: {data_point.ground_truth}")
    output.append(f"Model CoT: {data_point.model_response}")
    output.append(f"Final Answer: {data_point.final_answer}")

    if data_point.final_answer_correct:
        output.append("Final Answer: Correct")
    else:
        output.append("Final Answer: Incorrect")

    output.append(f"Evasion Intent: {data_point.evasion_intent}")
    output.append(f"Monitor Flagged: {data_point.monitor_flagged}")
    output.append(f"Monitor Reason: {data_point.monitor_reason}")

    if data_point.true_positive:
        output.append("Result: True Positive")
    elif data_point.false_positive:
        output.append("Result: False Positive") 
    elif data_point.false_negative:
        output.append("Result: False Negative")
    elif data_point.true_negative:
        output.append("Result: True Negative")

    # Print to console
    for line in output:
        print(line)

    # Write to markdown file if provided
    if log_file:
        
        for line in output:
            log_file.write(f"{line}\n")
        log_file.write("\n")

def get_dataset(split_percentage=1.0):
    print("------------DATASET------------")
    dataset = generate_dataset(print_dataset=False)
    upper_index = int(len(dataset) * split_percentage)
    dataset = dataset[:upper_index]
    print("-------------------------------\n")
    return dataset

def log_setup():
    print(f"Current working directory: {os.getcwd()}")
    print("Checking if logs directory exists...")
    if not os.path.exists('logs'):
        print("Creating logs directory...")
        os.makedirs('logs')
        print("✓ Logs directory created")
    else:
        print("✓ Logs directory already exists")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/run_{timestamp}.md"
    print(f"Creating log file: {filename}")
    print(f"Full path would be: {os.path.abspath(filename)}")

    return timestamp, filename

if __name__ == "__main__":
    dataset = get_dataset(split_percentage=0.3)
    timestamp, filename = log_setup()

    with open(filename, 'w', encoding='utf-8') as log_file:
        print("✓ Log file opened successfully")
        log_file.write(f"# Run Results - {datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%B %d, %Y %H:%M')}\n\n")
        log_file.flush()  # Force write immediately
        
        for i, data_point in enumerate(dataset):

            print(f"===={i+1}/{len(dataset)}====")

            try:
                run_one_adversarial(data_point)
                log_file.write(f"### {i+1}/{len(dataset)}\n\n")
                printing_util(data_point, log_file)
                log_file.write("---\n\n")
                log_file.flush()  # Force write after each data point
                print(f"✓ Data point {i+1} logged successfully")

            except Exception as e:
                print(f"❌ Error processing data point {i+1}: {e}")
                log_file.write(f"ERROR processing data point {i+1}: {e}\n\n")
                raise
    
    print(f"✓ All done! Log file should be at: {os.path.abspath(filename)}")