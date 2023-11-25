import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
import os
import json

data_path = './data/kokoro-speech-v1_3-large'
# output_path = os.path.abspath('./kokoro-speech')
output_path = './kokoro-speech'

with open(os.path.join(data_path, 'metadata.csv')) as fp:
    lines = fp.readlines()
#lines = lines[:100]

def trim_audio(infile, outfile, sr=22050, notrim=False):
    x, sr = librosa.load(infile, sr=sr, mono=True)
    if not notrim:
        y = []
        for s, e in librosa.effects.split(y=x):
            y.append(x[s:e])
        y = np.concatenate(y, axis=0)
    else:
        y = x
    sf.write(outfile, y, samplerate=sr)
    return y.shape[0] / sr

#os.makedirs(os.path.join(output_path, 'wavs'), exist_ok=True)

data = []
for line in tqdm(lines):
    parts = line.rstrip('\r\n').split('|')
    clip_id, text, norm_text = parts
    wav_inpath = os.path.join(data_path, 'wavs', f'{clip_id}.flac')
    wav_path = os.path.join(output_path, 'wavs', f'{clip_id}.flac')
    duration = trim_audio(wav_inpath, wav_path)
    data.append({
        "audio_filepath": wav_path,
        "text": text,
        'normalized_text': norm_text,
        "duration": duration,
        # "text": "yes monsieur",
        # "text_no_preprocessing": "Yes, monsieur.",
        # "text_normalized": "Yes, monsieur."}
    })

x = np.arange(len(data))
np.random.seed(1234)
np.random.shuffle(x)

def write_manifest(split, indices):
    with open(os.path.join(output_path, f"{split}_manifest.json"), 'wt') as f_out:
        for idx in indices:
            entry = data[idx]
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

write_manifest('train', x[600:])
write_manifest('val', x[500:600])
write_manifest('test', x[:500])
