from src.parsing import Parser
from src.logic_file import FunctionsDict
from llm_sdk import Small_LLM_Model
path_function_file = "data/input/functions_definition.json"
path_prompts_file = "data/input/function_calling_tests.json"
model = Small_LLM_Model() 
fun = FunctionsDict()


prompt = fun.prompt_list(path_prompts_file)[3]
list_func = fun.functions_list(path_function_file)
input_ids =  model.encode(fun.build_prompt(prompt, list_func)).tolist()[0]
fun.get_logits(input_ids)
