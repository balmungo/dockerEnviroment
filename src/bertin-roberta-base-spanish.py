import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = "../models/bertin-roberta-base-spanish"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Mover el modelo a GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

text = "La capital de España es <mask>."
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**inputs)

# Obtener la palabra predicha
predicted_token_id = outputs.logits[0, 5].argmax().item()  # Posición de <mask>
predicted_word = tokenizer.decode(predicted_token_id)
print(f"Texto completado: {text.replace('<mask>', predicted_word)}")

""" text = "El quechua es una lengua originaria."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Obtener el embedding de la última capa (útil para tareas de NLP)
embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
print(f"Embedding shape: {embeddings.shape}")  # Debería ser (768,) """