import torch
print(torch.__version__)
import torchaudio
print(torchaudio.list_audio_backends())
import sys
print(sys.version)
import json
print(json.dumps({"a":1,"b":2}))
""" from transformers import pipeline
print(pipeline('sentiment-analysis')('hugging face is the best'))
 """