import torch
import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torchaudio.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/whisper-base.en"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

def transcribe_audio_wav(audio_path):
 waveform, sample_rate = torchaudio.load(audio_path, format="wav")
 if waveform.shape[0] > 1:
  waveform = torch.mean(waveform, dim=0, keepdim=True)
 if sample_rate != 16000:
  resample = transforms.Resample(sample_rate, 16000)
  waveform = resample(waveform)

 input_features = processor(
  waveform.squeeze().numpy(), 
  sampling_rate=16000,
  return_tensors="pt"
  )
 attention_mask = torch.ones_like(input_features.input_features)
 input_features["attention_mask"] = attention_mask
 input_features = {k: v.to(device) for k, v in input_features.items()}
 
 model.eval()
 with torch.no_grad():
  predicted_ids = model.generate(**input_features,max_new_tokens=400,suppress_tokens=[])
 transcription = processor.batch_decode(predicted_ids,skip_special_tokens=True)
 return transcription[0]

audio_path = "./sample/input_audio.wav"
generated_text = transcribe_audio_wav(audio_path)

print("salida escrita: "+generated_text+" fin.")