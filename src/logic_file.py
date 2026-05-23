import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_sdk'))
from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    print("Loading model...")
    model = Small_LLM_Model()

    prompt = "Greet shrek"
    input_ids = model.encode(prompt).tolist()[0]
    print(f"Input IDs: {input_ids}")

    logits = model.get_logits_from_input_ids(input_ids)
    print(f"Logits length: {len(logits)}")
    print(f"Max logit value: {max(logits)}")
