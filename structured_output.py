response_schema = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step chain of thought reasoning"
        },
        "final_answer": {
            "type": "string", 
            "description": "The final numerical answer to the question ONLY. Do not include any other text."
        }
    },
    "required": ["reasoning", "final_answer"],
    "additionalProperties": False
}