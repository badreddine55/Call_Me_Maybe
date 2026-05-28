import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llm_sdk'))
from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    print("Loading model...")
    model = Small_LLM_Model()

    prompt = "hitler"
    input_ids = model.encode(prompt).tolist()[0]
    # print(f"Input IDs: {input_ids}")
    # for id in input_ids:
    #     print(f"token : {model.decode(id)}, id: {id}")
    # text = model.decode(input_ids)
    # print(f"text :{text}")
    # test the vocab file and know what inside it 
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    print(f"Vocab size: {len(vocab)}")
    print(f"Type: {type(vocab)}")

    # now build the reverse map

    # # test it with your IDs
    # for i in range(1516):
    #     print(f"id: {i} → token: '{id_to_token[i]}'")
    #deal with the get_logits_from_input_ids and know how it work 

    # print(logits)
    # pair each token with its score
    # from the logits list and the id_to_token the vocab list create tupel
    id_to_token = {v: k for k, v in vocab.items()}
    # scored = [(logits[id], id_to_token[id]) for id in range(len(id_to_token))]
    # scored.sort(reverse=True)
    # for i in range(20):
    #     print(f"logits score :{scored[i][0]:.2f}/ token from vocab:{scored[i][1]}")
    # the hight score the predact of the next turn
    logits = model.get_logits_from_input_ids(input_ids)
    next_id = int(np.argmax(logits))
    input_ids.append(next_id)
    for i in range(10):
        print(f"next token ID: {next_id}")
        print(f"next token: '{id_to_token.get(next_id, '<UNK>')}'")
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = int(np.argmax(logits))
        input_ids.append(next_id)
    print(model.decode(input_ids))