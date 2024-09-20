prompts = [
    "Hello, my name is",
]
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2)

#llm = LLM(model="/app/model")
#llm = LLM(model="/data/llama-2-7b-chat-hf", tensor_parallel_size=2)
llm = LLM(model="/data/llama2-70b-chat",  enforce_eager=True, tensor_parallel_size=4)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


