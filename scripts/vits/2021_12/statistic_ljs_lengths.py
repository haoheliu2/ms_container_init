import numpy as np 
import librosa 
import os  

root_dir = "/Users/liuhaohe/Listener_test/vits/LJSpeech-1.1/wavs"
length = []
from tqdm import tqdm 
for file in tqdm(os.listdir(root_dir)):
    path = os.path.join(root_dir, file)
    x,_ = librosa.load(path)
    length.append(x.shape[0])
print(np.mean(length))