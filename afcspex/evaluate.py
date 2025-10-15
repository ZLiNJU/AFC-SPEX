import os
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

import torchaudio
import argparse
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import FileDataset as Dataset
from torchmetrics import ScaleInvariantSignalDistortionRatio
from pesq import pesq
from pystoi import stoi
import pandas as pd

def main(args, config):

    si_sdr = ScaleInvariantSignalDistortionRatio()
    sample_rate = config.sample_rate
    enh_folder = config.infer.enh_folder
    data_num = config.test_dataset.data_num
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(**config['test_dataset'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset, **config['test_dataloader'])
    
    total_si_sdr = 0
    total_pesq = 0
    total_stoi = 0

    for step, batch_data in enumerate(tqdm(dataloader, ncols=123), 1):

        target = batch_data['target_speaker_at_mic'].permute(0, 2, 1)
        file_cnt = batch_data['file_cnt']
        
        for i in range(target.shape[0]):

            uid = str(file_cnt[i].item())
            enh_path = os.path.join(enh_folder, f"{uid}_enh.wav")
            enhanced, _ = torchaudio.load(enh_path)
            enhanced = enhanced.squeeze()
            total_si_sdr += si_sdr(enhanced.cpu(), target[i, :, 0].cpu())
            total_pesq += pesq(fs=sample_rate, ref=target[i, :, 0].cpu().numpy(), deg=enhanced.cpu().numpy(), mode='nb')
            total_stoi += stoi(target[i, :, 0].cpu().numpy(), enhanced.cpu().numpy(), fs_sig=sample_rate)

    aver_si_sdr = total_si_sdr / data_num
    aver_pesq = total_pesq / data_num
    aver_stoi = total_stoi / data_num

    results = pd.DataFrame({
        'Metric': ['SI-SDR', 'PESQ', 'STOI'],
        'Average': [aver_si_sdr, aver_pesq, aver_stoi]
    })
    save_path = os.path.join(enh_folder, 'evaluation_results.csv')
    results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    print(f"Average SI-SDR: {aver_si_sdr:.2f}")
    print(f"Average PESQ: {aver_pesq:.2f}")
    print(f"Average STOI: {aver_stoi:.2f}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-IC', '--infer_config', default='configs/infer_config.yaml')
    parser.add_argument('-MC', '--model_config', default='configs/model_config.yaml')
    parser.add_argument('-D', '--device', default='0', help='The index of the device')

    args = parser.parse_args()
    conf_infer = OmegaConf.load(args.infer_config)
    conf_model = OmegaConf.load(args.model_config)
    config = OmegaConf.merge(conf_infer, conf_model)

    main(args, config)