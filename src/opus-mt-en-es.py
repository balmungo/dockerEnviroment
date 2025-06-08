import torch
import sys
from transformers import MarianMTModel, MarianTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/opus-mt-en-es"

try:
 tokenizer = MarianTokenizer.from_pretrained(model_path,use_fast=False,local_files_only=True)
 model = MarianMTModel.from_pretrained(model_path,local_files_only=True)
 model.to(device)
except Exception as e:
 print(f"Error al cargar el modelo: {e}")
 exit()


def translate(textEnglish):
 inputs = tokenizer(textEnglish,return_tensors="pt",padding=True).to(device)
 traduccion = model.generate(**inputs,max_new_tokens=100) 
 return tokenizer.decode(traduccion[0],skip_special_tokens=True)

English_text = sys.argv[1]
Spain_text = translate(English_text)

print(Spain_text)