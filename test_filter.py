import json
from transformers import AutoTokenizer, AutoModelForCausalLM


with open("./LLaMA-Factory/data/sft_dataset.json", "r") as f:
    data = json.load(f)

example = data[0]
instruction = example["instruction"]
input = example["input"]

messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    },
    {
        "role": "user",
        "content": f"{instruction}\n{input}"
    }
]


model_name = "nandansarkar/qwen3_0-6B_filter_13_epochs"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# no thinking mode
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False   
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating output...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

output_text = tokenizer.decode(
    generated_ids[0][len(model_inputs.input_ids[0]):],
    skip_special_tokens=True
)


print(output_text)
