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

F0_PATH = "/home/v-haoheliu/data/LJSpeech-1.1-fs2/f0/"  
source_file = "filelists/ljs_audio_text_test_filelist.txt.cleaned"
target_file = "filelists/ljs_audio_text_test_pitch_filelist.txt.cleaned"
res = []
for each in read_list(source_file):
    fname = each.split("|")[0]
    new_path = os.path.join(F0_PATH, os.path.basename(fname))[:-4]+".npy"
    res.append(each+"|"+new_path)

write_list(res, target_file)