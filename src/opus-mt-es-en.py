import torch
import sys
from transformers import MarianMTModel, MarianTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/opus-mt-es-en"

try:
 tokenizer = MarianTokenizer.from_pretrained(model_path,local_files_only=True)
 model = MarianMTModel.from_pretrained(model_path,local_files_only=True)
 model.to(device)
except Exception as e:
 print(f"Error al cargar el modelo: {e}")
 exit()

def translate(textSpain):
 inputs = tokenizer(textSpain,return_tensors = "pt",padding=True).to(device)
 traduccion = model.generate(**inputs,max_new_tokens=100)
 return tokenizer.decode(traduccion[0],skip_special_tokens=True)

Spain_text = sys.argv[1]
English_text = translate(Spain_text)

print(English_text)
 