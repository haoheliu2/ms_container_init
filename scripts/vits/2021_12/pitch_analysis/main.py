import pyworld as pw
import numpy as np 
import matplotlib.pyplot as plt
import librosa
import torch  
import os 

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0

# target = "/Users/liuhaohe/Listener_test/vits/LJSpeech-1.1/wavs"
# source = "/Users/liuhaohe/Listener_test/vits/output_1160000"
# target = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-vits-and-z_p_pp/vits"
target = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-duration/baseline_gt"
source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-duration/baseline_gt" # -1.3599075e-09 -5.643797194126207e-10 -1.7147679e-08
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-vits-and-z_p_pp/z_p_pp" # 11.075687 12.109407970187501 3.6561332
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-vits-and-z_p_pp/vits" # 14.108599 13.254652084494023 2.9181716
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-04-vits-and-ft-lul/ft-lul" # 11.233016 13.463750108325694 3.2629306
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-06-vits-and-ft-lul-reverse/ft_lul_reverse" # 11.565763 11.708196476418898 0.26347193
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-dp-vits-and-dp-vits-z-pp/dp-vits" # 11.896502 12.808223713833954 2.9892254
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-dp-vits-and-dp-vits-z-pp/dp-vits-z-pp" # 11.155307 12.762083907342864 3.3576374
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-z/gt_z" # 4.760997 7.358288014844757 2.036166
# source = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-duration/gt_duration" # 9.420631 10.73848022489989 2.303613
sources = [ 
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-duration/baseline_gt",
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-vits-and-z_p_pp/z_p_pp", # 11.075687 12.109407970187501 3.6561332
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-vits-and-z_p_pp/vits", # 14.108599 13.254652084494023 2.9181716
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-04-vits-and-ft-lul/ft-lul", # 11.233016 13.463750108325694 3.2629306
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-06-vits-and-ft-lul-reverse/ft_lul_reverse", # 11.565763 11.708196476418898 2.9437654
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-dp-vits-and-dp-vits-z-pp/dp-vits", # 11.896502 12.808223713833954 2.9892254
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-03-dp-vits-and-dp-vits-z-pp/dp-vits-z-pp", # 11.155307 12.762083907342864 3.3576374
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/exps/ljs_exp10_ft_z_z_p_x-2722000", # 10.876444 11.79602339511501 3.4621634
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/exps/exp10_pitch_finetuning", # 13.636864 13.090485494571903 3.5993786
    "/Users/liuhaohe/Downloads/temp2", # 11.227406 13.183360017788143 3.3564
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-z/gt_z", # 4.760997 7.358288014844757 2.036166
    # "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-and-gt-duration/gt_duration" # 9.420631 10.73848022489989 2.303613
]

SOURCE_PATH="."
ID="temp"

def get_pitch(wav_data, mel):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    _f0, t = pw.dio(wav_data.astype(np.double), 22050,
                    frame_period=256 / 22050 * 1000)
    f0 = pw.stonemask(wav_data.astype(np.double), _f0, t, 22050)  # pitch refinement
    delta_l = len(mel) - len(f0)
    # assert np.abs(delta_l) <= 20, delta_l
    if delta_l > 0:
        f0 = np.concatenate([f0] + [f0[-1]] * delta_l)
    f0 = f0[:len(mel)]
    # pitch_coarse = f0_to_coarse(f0) + 1
    return f0, None

def pitch(file):
    def process_f0(f0):
        pitch_mean = 127.39336246019445
        pitch_std = 109.13872671910696
        f0_ = (f0 - pitch_mean) / pitch_std
        f0_[f0 == 0] = np.interp(np.where(f0 == 0)[0], np.where(f0 > 0)[0], f0_[f0 > 0])
        uv = f0 == 0
        return f0_, uv
    wav, sr = librosa.load(file)
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80).T
    f0,_ = get_pitch(wav, mel)
    return process_f0(f0)

def visualize_pitch(pitch1, pitch2, fname): 
    plt.figure()
    plt.plot(pitch1)
    plt.plot(pitch2)
    plt.legend(["estimated","target"])
    plt.xlabel("frames")
    plt.xlabel("normalized value")
    # plt.show()
    plt.savefig(os.path.join(SOURCE_PATH,fname+".png"))
    plt.close()
    
def visualize_img(img_est, img_target, fname): 
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow(img_est, aspect='auto')
    plt.xlabel("delta")
    plt.ylabel("frames")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(img_target, aspect='auto')
    plt.xlabel("delta")
    plt.ylabel("frames")
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join(SOURCE_PATH,fname+".png"))  
    plt.close()  
    
from soft_dtw import SoftDynamicTimeWarping
soft_dtw = SoftDynamicTimeWarping(penalty=0.1,
                                  gamma=0.01,
                                  bandwidth=120,
                                  dist_func="manhattan",
                                  average_alignment_save_path=".",
                                  device="cpu")

def soft_dtw_dist(pitch1, pitch2):
    return soft_dtw(torch.tensor(pitch1)[None,...,None].float(), torch.tensor(pitch2)[None,...,None].float(),
                    torch.ones((pitch1.shape[0])).bool()[None,...], torch.ones((pitch2.shape[0])).bool()[None,...],
                    save_alignment_flag=True, global_steps=1000)
    
def soft_dtw_delta_dist(pitch1, pitch2, fname, n=20):
    pitch1 = torch.tensor(pitch1)[None,...,None]
    pitch2 = torch.tensor(pitch2)[None,...,None]
    diff1, diff2 = None, None
    for i in range(1,n):
        t1 = pitch1[:,i:,:] - pitch1[:,:-i,:]
        t2 = pitch2[:,i:,:] - pitch2[:,:-i,:]
        # visualize_pitch(t1[0,:,0].numpy(), t2[0,:,0].numpy())
        if(diff1 is None):
            diff1 = t1
        else: 
            diff1 = diff1[:,:t1.size(1),:]
            diff1 = torch.cat([diff1,t1],dim=-1)
        if(diff2 is None): 
            diff2 = t2
        else: 
            diff2 = diff2[:,:t2.size(1),:]
            diff2 = torch.cat([diff2,t2],dim=-1)
    mask1 = torch.ones((diff1.size(1))).bool()[None,...]
    mask2 = torch.ones((diff2.size(1))).bool()[None,...]
    visualize_img(diff1[0,...].numpy(),diff2[0,...].numpy(), fname=fname)
    return soft_dtw(diff1, diff2, mask1, mask2, save_alignment_flag=False, global_steps=1000)
       
def main(): 
    import os 
    from tqdm import tqdm
    linear = torch.nn.Linear(1,1)
    global ID, SOURCE_PATH, sources, target
    for source in sources:
        SOURCE_PATH = source
        pitch_all = []
        delta_all = []
        uv_all = []
        for i, each in enumerate(tqdm(os.listdir(source))): 
            # if(i > 5): break
            # print(each)
            ID=each
            if(each[-4:]!=".wav"): continue 
            base_name = each.split(".")[0]
            source_file = os.path.join(source, each)
            target_file = os.path.join(target, base_name+".wav")
            f0_source, uv_source = pitch(source_file)
            f0_target, uv_target = pitch(target_file)
            pitch_dist, pitch_E = soft_dtw_dist(f0_source, f0_target)
            delta,delta_E = soft_dtw_delta_dist(f0_source, f0_target, fname="delta_"+each)
            uv, uv_E = soft_dtw_dist(uv_source, uv_target)
            print(f0_source.shape, f0_target.shape, pitch_E.size())
            pitch_all.append(pitch_dist)
            delta_all.append(delta)
            uv_all.append(uv)
            
            # f0_source = np.matmul(f0_source[None,...],pitch_E[0].numpy())[0,...]
            # print(uv_source.shape)
            # visualize_pitch(uv_source, uv_target, fname="uv_"+each)
            visualize_pitch(f0_source, f0_target, fname="pitch_"+each)
            # break
        print(np.mean(pitch_all), np.mean(delta_all), np.mean(uv_all))
        
if __name__ == "__main__":
    main()