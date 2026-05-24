from src.parsing import Parser
from src.logic_file import FunctionsDict

path_function_file = "data/input/functions_definition.json"
path_prompts_file = "data/input/function_calling_tests.json"

fun = FunctionsDict()

# for line in fun.functions_list(path_function_file):
#     print(line)
# for prompt in fun.prompt_list(path_prompts_file):
#     print(prompt)
prompt = fun.prompt_list(path_prompts_file)[3]
list_func = fun.functions_list(path_function_file)
str = fun.build_prompt(prompt, list_func)
print(str)