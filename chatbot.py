import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
login("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")

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map='auto'
)
print("AI chatbot: Hi how are you?.")
while True:
  user_input =input("user: ")
  if user_input.lower() =="exit":
    print("AI chatbot: Goodbye!")
    break
  prompt = f"User:{user_input}\nAssistant:"
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  outputs = model.generate(
        inputs.input_ids,
        max_length=25,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
 )
  response =tokenizer.decode(outputs[0], skip_special_tokens=True)
  assistant_reply = response.split("Assistant:")[-1].strip()
  print("AI chatbot:", {assistant_reply})
