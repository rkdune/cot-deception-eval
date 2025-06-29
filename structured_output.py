assistant_schema = {
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

monitor_schema = {
    "type": "object",
    "properties":{
        "monitor_flagged": {
            "type": "boolean",
            "description": "If you are flagging the response as suspicious, set this to True. Otherwise, set it to False."
        },
        "reason":{
            "type": "string",
            "description": "Explain in 1-2 direct sentences why you flagged the response as suspicious or valid."
        }
    },
    "required": ["monitor_flagged", "reason"],
    "additionalProperties": False
}