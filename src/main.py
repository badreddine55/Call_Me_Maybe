from src.parsing import Parser
from src.logic_file import FunctionsDict
path_function_file = "data/input/functions_definition.json"
path_prompts_file = "data/input/function_calling_tests.json"
fun = FunctionsDict()


# prompt = fun.prompt_list(path_prompts_file)[8]
# list_func, functions_names = fun.functions_list(path_function_file)
# input_ids =  fun.model.encode(fun.build_prompt(prompt, list_func)).tolist()[0]
# fun.get_logits(input_ids)
# function_name_tokens = fun.extract_function_name(functions_names, input_ids)
# function_name = fun.model.decode(function_name_tokens)
# fun.extract_function_params(function_name, input_ids)

for prompt in fun.prompt_list(path_prompts_file):
    list_func, functions_names = fun.functions_list(path_function_file)
    input_ids = fun.model.encode(fun.build_prompt(prompt, list_func)).tolist()[0]
    function_name_tokens = fun.extract_function_name(functions_names, input_ids)
    function_name = fun.model.decode(function_name_tokens)
    fun.extract_function_params(function_name, input_ids)
    print("---")


