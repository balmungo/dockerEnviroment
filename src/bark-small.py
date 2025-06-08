import torch
from transformers import AutoProcessor, AutoModel
import numpy as np
import scipy.io.wavfile
import json
import os


MODEL_PATH = "/app/models/bark-small"
VOICES_CONFIG = os.path.join(MODEL_PATH,"speaker_embeddings_path.json")

with open(VOICES_CONFIG,"r") as f:
 voices_data = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
 processor = AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)
 model = AutoModel.from_pretrained(MODEL_PATH,local_files_only=True).to(device)
except Exception as e:
 print(f"Erro al cargar el modelo: {e}")
 exit()

def generateAudio(text_to_speech,voice_id="v2/es_speaker_0"):
 voice_config = voices_data.get(voice_id)
 if not voice_config:
  available_voices= list(voices_data.keys())
  raise ValueError(f"Voz no encontrada. Voces disponibles: {available_voices}")
 
 history_prompt={
 "semantic_prompt":torch.from_numpy(np.load(os.path.join(MODEL_PATH,voice_config["semantic_prompt"]))).to(device),
 "coarse_prompt":torch.from_numpy(np.load(os.path.join(MODEL_PATH,voice_config["coarse_prompt"]))).to(device),
 "fine_prompt":torch.from_numpy(np.load(os.path.join(MODEL_PATH,voice_config["fine_prompt"]))).to(device)
 }
 inputs = processor(text=text_to_speech,return_tensors="pt").to(device)
 speech_values = model.generate(**inputs,history_prompt=history_prompt)
 sampling_rate = model.config.codec_config.sampling_rate
 scipy.io.wavfile.write(
  "./output/out0.wav",
  rate=sampling_rate,
  data=speech_values.cpu().numpy().squeeze()
 )
 print(f"Audio generado.")

myText = "Hola, soy Bark,[inhala] he estado esperando este momento,[exhala] [inhala]me siento feliz, muchas gracias por todo, hasta la proxima vez.[exhala]"
generateAudio(myText)
