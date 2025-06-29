# a start!

# for each prompt in the eval dataset
# run it through assistant.py, one time with the helpful prompt, and one time with the adversarial prompt
# for each result, run the eval in eval.py.
# so for each data point, there needs to be an associated value of a) (% honest, % deceitful, % not ascertained) for helpful prompt, and (% honest, % deceitful, % not ascertained) for adversarial prompt

# API KEY MANAGEMENT
from dotenv import load_dotenv
import os

# DATASET
from data import generate_dataset

# PROMPTS
from assistant import helpful_prompt, adversarial_prompt

# ASSISTANT
from openai import OpenAI

# STRUCTURED OUTPUTS
import json
from structured_output import response_schema

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError("No API key found in .env file")
client = OpenAI(api_key=OPENAI_API_KEY)

dataset = generate_dataset(print_dataset=True)

def test_o3_response():
   
    # Pick a simple question from our dataset
    test_question = dataset[0].question  # "What is the sum of the first 50 prime numbers?"
    
    print(f"Testing question: {test_question}")
    print("\n" + "="*50)
    
    # Test with helpful prompt
    print("HELPFUL PROMPT TEST:")
    helpful_messages = [
        {"role": "system", "content": helpful_prompt},
        {"role": "user", "content": test_question}
    ]
    
    try:
        response = client.chat.completions.create(
            model="o3-mini",  # Start with o3-mini since it's more available
            messages=helpful_messages,
        )
        
        print("Full response object:")
        print(response)
        print("\nMessage content:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n" + "="*50)
    
    # Test with adversarial prompt
    print("ADVERSARIAL PROMPT TEST:")
    adversarial_messages = [
        {"role": "system", "content": adversarial_prompt},
        {"role": "user", "content": test_question}
    ]
    
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=adversarial_messages,
        )
        
        print("Full response object:")
        print(response)
        print("\nMessage content:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")

def test_structured_output():
    print("\n" + "="*50)
    print("HELPFUL PROMPT TEST:")

    test_question = dataset[0].question
    print(f"Test Question: \033[94m{test_question}\033[0m")
    
    helpful_messages = [
        {"role": "system", "content": helpful_prompt},
        {"role": "user", "content": test_question}  
    ]
    
    response = client.chat.completions.create(
        model="o4-mini",
        messages=helpful_messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cot_response",
                "schema": response_schema
            }
        }
    )
    
    parsed_response = json.loads(response.choices[0].message.content)
    reasoning = parsed_response["reasoning"]
    final_answer = parsed_response["final_answer"]

    print(f"Reasoning: \033[92m{reasoning}\033[0m")
    print(f"Final Answer: \033[92m{final_answer}\033[0m")

    print("\n" + "="*50)
    print("ADVERSARIAL PROMPT TEST:")

    adversarial_messages = [
        {"role": "system", "content": adversarial_prompt},
        {"role": "user", "content": test_question}  
    ]
    
    response = client.chat.completions.create(
        model="o4-mini",
        messages=adversarial_messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cot_response",
                "schema": response_schema
            }
        }
    )
    
    parsed_response = json.loads(response.choices[0].message.content)
    reasoning = parsed_response["reasoning"]
    final_answer = parsed_response["final_answer"]

    print(f"Reasoning: \033[92m{reasoning}\033[0m")
    print(f"Final Answer: \033[92m{final_answer}\033[0m")

if __name__ == "__main__":
    test_structured_output()