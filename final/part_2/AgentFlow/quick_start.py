# Import the solver
from agentflow.agentflow.solver import construct_solver

# Set the LLM engine name
llm_engine_name = "ollama-qwen2.5:7b"
# llm_engine_name = "dashscope" # you can use "dashscope" as well, to use the default API key in the environment variables qwen2.5-7b-instruct

# Construct the solver
# - model_engine: set all 4 components (planner_main, planner_fixed, verifier, executor)
#   to "trainable" so they all use llm_engine_name instead of the default "gpt-4o"
# - enabled_tools / tool_engine: list only tools that don't need unavailable API keys,
#   and use "self" so each tool also gets initialized with llm_engine_name (ollama)
solver = construct_solver(
    llm_engine_name=llm_engine_name,
    model_engine=["trainable", "trainable", "trainable", "trainable"],
    enabled_tools=["Base_Generator_Tool"],
    tool_engine=["self"],
)

# Solve the user query
output = solver.solve("What is the capital of France?")
print(output["direct_output"])