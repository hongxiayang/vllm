
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4, distributed_executor_backend="mp")
output = llm.generate("San Franciso is a")

