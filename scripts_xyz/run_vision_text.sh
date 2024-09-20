llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    image_input_type="pixel_values",
    image_token_id=32000,
    image_input_shape="1,3,336,336",
    image_feature_size=576,
)
rompt = "<image>" * 576 + (
    "\nUSER: What is the content of this image?\nASSISTANT:")

# Load the image using PIL.Image
image = ...

"role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": ImagePixelData(image),
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

