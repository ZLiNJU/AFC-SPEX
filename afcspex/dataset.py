import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Literal

class FileDataset(Dataset):
    def __init__(self, dir_path: str, data_num: int = 1000, 
                 sample_rate: int = 16000, mode: Literal["train", "val", "test"] = "train"):

        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found at {dir_path}")
        self.dir_path = os.path.abspath(dir_path)

        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['train', 'val', 'test']")
        self.mode = mode

        self.fs = sample_rate
        
        self.rir_path = os.path.join(self.dir_path, f"rir/{self.mode}/")
        self.audio_path = os.path.join(self.dir_path, f"audio/{self.mode}/")
        if not os.path.isdir(self.rir_path):
            raise FileNotFoundError(f"RIR directory not found at {self.rir_path}")
        if not os.path.isdir(self.audio_path):
            raise FileNotFoundError(f"Audio directory not found at {self.audio_path}")

        try:
            actual_files = len([name for name in os.listdir(self.audio_path) if name.startswith('tgt_reverb_') and name.endswith('.wav')])
        except OSError:
            actual_files = 0
        
        if actual_files == 0:
            raise RuntimeError(f"No data files found in {self.audio_path}")

        self.num = min(data_num, actual_files)
        print(f"Found {actual_files} files, using {self.num} files for dataset in '{self.mode}' mode.")

    def audioread(self, wav_path):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        wav, sr = torchaudio.load(wav_path)
        if sr != self.fs:
            wav = torchaudio.transforms.Resample(sr, self.fs)(wav)
        wav = wav.squeeze()
        return wav

    def __getitem__(self, item):
        try:

            if self.mode == 'test': # Only in test mode, closed-loop simulation is used, so load RIR
                rir_file = os.path.join(self.rir_path, f'{item}.npy')
                rir = np.load(rir_file)
                rir = torch.from_numpy(rir).float()
            else:
                rir = ()

            file_cnt = item

            target_speaker_at_mic = self.audioread(os.path.join(self.audio_path, f'tgt_reverb_{item}.wav'))
            interf_speaker_at_mic = self.audioread(os.path.join(self.audio_path, f'noise_interf_{item}.wav'))

            mic_rec_wfb_file = os.path.join(self.audio_path, f'rec_wfb_{item}.npy')
            mic_rec_wfb = np.load(mic_rec_wfb_file)
            mic_rec_wfb = torch.from_numpy(mic_rec_wfb).float()

            afc_res_file = os.path.join(self.audio_path, f'afc_res_{item}.npy')
            afc_res = np.load(afc_res_file)
            afc_res = torch.from_numpy(afc_res).float()        

            return {
                    "target_speaker_at_mic": target_speaker_at_mic,
                    "interf_speaker_at_mic": interf_speaker_at_mic,
                    "mic_rec_wfb": mic_rec_wfb,
                    "afc_res": afc_res,
                    "rir": rir,
                    "file_cnt": file_cnt
                    }

        except FileNotFoundError as e:
            print(f"Warning: File not found for item {item}. {e}. Skipping.")
            return None
        
        except Exception as e:
            print(f"Warning: Error loading data for item {item}. {e}. Skipping.")
            return None

    def __len__(self):
        return self.num

if __name__ == "__main__":
    
    config = OmegaConf.load('/home/nis/ze.li/ResearchProject/AFC/rebuild_AFCSPEX/configs/trainer_config.yaml')

    try:
        invalid_dataset = FileDataset(config.trainer.data_set_path, mode="invalid_mode")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    data_set = FileDataset(**config.train_dataset)
    data_loader = DataLoader(dataset=data_set, **config.train_dataloader)

    for batch_idx, batch_data in enumerate(data_loader):
        print(f"Batch {batch_idx}, File cnt: {batch_data['file_cnt'].tolist()}")
        print(f"target_speaker shape: {batch_data['target_speaker_at_mic'].shape}")