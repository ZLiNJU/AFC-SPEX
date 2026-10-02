"""Load the prepared speech, interference and room responses."""

import glob
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class FileDataset(Dataset):
    def __init__(self, dir_path, num=None, reverse=False, fs=16000, mode="train"):
        self.fs = fs
        self.num = num
        self.mode = mode

        audio_path = os.path.join(dir_path, "audio", mode)
        self.tgt_reverb_list = sorted(glob.glob(audio_path + "/*tgt_reverb*.wav"))
        self.noise_interf_list = sorted(glob.glob(audio_path + "/*noise_interf*.wav"))

        rir_path = os.path.join(dir_path, "rir", mode)
        self.rir_list = sorted(glob.glob(rir_path + "/*.npy"))

        if reverse:
            self.tgt_reverb_list.reverse()
            self.noise_interf_list.reverse()
            self.rir_list.reverse()

        if num is not None:
            self.tgt_reverb_list = self.tgt_reverb_list[:num]
            self.noise_interf_list = self.noise_interf_list[:num]
            self.rir_list = self.rir_list[:num]

    def audioread(self, wav_path):
        wav, _ = torchaudio.load(wav_path)
        return wav.squeeze()

    def __getitem__(self, item):
        tgt_reverb = self.audioread(self.tgt_reverb_list[item]).T
        noise_interf = self.audioread(self.noise_interf_list[item]).T
        rir = torch.from_numpy(np.load(self.rir_list[item])).float()

        file_name = os.path.basename(self.tgt_reverb_list[item])
        file_num = file_name.split("_")[2].split(".")[0]
        return tgt_reverb, noise_interf, rir, file_num

    def __len__(self):
        return len(self.tgt_reverb_list)
