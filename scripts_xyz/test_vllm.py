from vllm import LLM, SamplingParams

def test():
    prompts = [
            "San Franciso is a",
    #     "A story about vLLM:",
        "DeepSpeed is a machine learning library that deep learning practitioners should use for what purpose",
        "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=80)
    #sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=200)

    #llm = LLM(model="/app/model")
    ##llm = LLM(model="/data/llama-2-7b-chat-hf", enable_chunked_prefill=True, enforce_eager=True, tensor_parallel_size=1)
    #llm = LLM(model="/data/llama-2-7b-chat-hf", tensor_parallel_size=2, distributed_executor_backend="mp" )
    #llm = LLM(model="/data/llama-2-7b-chat-hf" )
    #llm = LLM(model="/data/llama2-70b-chat",  tensor_parallel_size=4, distributed_executor_backend="mp")
    llm = LLM(model="/data/Mixtral-8x7B-v0.1",  tensor_parallel_size=4, distributed_executor_backend="mp")

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    test()


