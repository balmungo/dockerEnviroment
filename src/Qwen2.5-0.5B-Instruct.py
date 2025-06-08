import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/Qwen2.5-0.5B-Instruct"

try:
 tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
 model = AutoModelForCausalLM.from_pretrained(
  model_path,
  torch_dtype=torch.float16,  # O usa "auto"
  device_map="auto",
  trust_remote_code=True
 )
except Exception as e:
 print(f"Error al cargar el modelo: {e}")
 exit()

prompt = "hola, ¿como esta el clima?."
messages = [
 {"role": "system", "content": "You are Qwen, You are a helpful assistant."},
 {"role": "user", "content": prompt}
]

# Corregido aquí
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100)

# Cortamos solo la respuesta generada, quitando el prompt
generated_ids = [
 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
