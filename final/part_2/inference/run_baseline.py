from agentflow.solver import construct_solver

llm_engine_name = "ollama-qwen2.5:7b"

solver = construct_solver(llm_engine_name=llm_engine_name)

output = solver.solve("What is the capital of France?")
print(output["direct_output"])