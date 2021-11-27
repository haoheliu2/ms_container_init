import numpy as np     
import os
import torch
import os

def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def write_list(list, fname):
    with open(fname,'w') as f:
        for word in list:
            f.write(word)
            f.write('\n')
        
pitch_mean = 127.39336246019445
pitch_std = 109.13872671910696
        
def process_f0(f0):
    f0_ = (f0 - pitch_mean) / pitch_std
    f0_[f0 == 0] = np.interp(np.where(f0 == 0)[0], np.where(f0 > 0)[0], f0_[f0 > 0])
    uv = (torch.FloatTensor(f0) == 0).float()
    f0 = f0_
    f0 = torch.FloatTensor(f0)
    return f0, uv

def get_pitch(pitchpath):
    pitch = np.load(pitchpath).astype(np.float32)[None, :-1] # todo remove an extra frame
    return process_f0(pitch)

from tqdm import tqdm

for file in os.listdir(os.path.join("filelists")): 
    print(file)
    if("pitch" in file):
        for each in tqdm(read_list(os.path.join("filelists",file))): 
            pitch_path = each.split("|")[-1]
            f0, uv = get_pitch(pitch_path)
            np.save("interp_"+os.path.basename(pitch_path),f0)
            np.save("uv_"+os.path.basename(pitch_path),uv)
            

        