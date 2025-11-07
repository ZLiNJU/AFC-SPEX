import os
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

import argparse
import torch
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
import importlib
from dataset import FileDataset as Dataset

def main(args, config):
    
    model_name = config.infer.model_name
    Model = dynamic_import(config)
    sample_rate = config.sample_rate
    enh_folder = config.infer.enh_folder
    os.makedirs(enh_folder, exist_ok=True)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset(**config['test_dataset'])
    dataloader = torch.utils.data.DataLoader(dataset=dataset, **config['test_dataloader'])

    model = Model(**config[model_name]['infer_init_args']).to(device)
    checkpoint = torch.load(config.infer.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.inference_mode():
        for step, batch_data in enumerate(tqdm(dataloader, ncols=123), 1):

            # For all LSTM based models and kalman filters, 
            # remember to call reset_states() between different sequences
            # for the fact that all hidden states are stored in the model
            # and filter tap weights are also stored in the model

            model.reset_states() 

            target = batch_data['target_speaker_at_mic'].to(device).permute(0, 2, 1)
            interf = batch_data['interf_speaker_at_mic'].to(device).permute(0, 2, 1)
            rir = batch_data['rir'].to(device)
            file_cnt = batch_data['file_cnt']

            rir = rir[:, :config.pbfd_config.rir_len, :] / 5 # TODO
            output = closed_loop_simulation(model = model, target = target, interf = interf, 
                                            rir = rir, config = config, device = device)
            
            for i in range(output.shape[0]):
                enhanced = output[i].cpu().numpy()
                uid = str(file_cnt[i].item())
                enh_path = os.path.join(enh_folder, f"{uid}_enh.wav")
                sf.write(enh_path, enhanced, sample_rate)

    config_save_path = os.path.join(config.infer.enh_folder, "infer_config.yaml")
    OmegaConf.save(config, config_save_path)

def dynamic_import(config):
    """
    Args:
        config: OmegaConf.merge(conf_infer, conf_model)
    Returns:
        class object: The imported class object.
    """
    class_name = config.infer.model_name

    if class_name not in config:
        raise KeyError(f" cannot find '{class_name}' ")
    
    class_path = config[class_name]._target_

    try:
        class_info = importlib.import_module(class_path)

        OutPutClass = getattr(class_info, class_name)

        return OutPutClass

    except ImportError:
        raise ImportError(f"cannot import '{class_path}'. Please check if the module path is correct.")
    except AttributeError:
        raise AttributeError(f"cannot find class '{class_name}' in '{class_path}'. Please check if the class name is correct.")

def closed_loop_simulation(model: torch.nn.Module, target: torch.Tensor, 
                           interf: torch.Tensor, rir: torch.Tensor, 
                           config, device):
    '''
    Closed-loop simulation, audio is processed frame by frame

    :param target: [n_batch, n_time, n_mic]
    :param interf: [n_batch, n_time, n_mic]
    :param rir: [n_batch, rir_len, n_mic]
    :return: [n_batch, n_time]
    '''

    model_name = config.infer.model_name
    delay = config.delay
    gain = config.gain
    maxamp = config.maxamp
    n_batch, n_time, n_mic = target.shape
    hop_len = config.pbfd_config.hop_len
    fft_len = config.pbfd_config.fft_len
    n_overlap = config.pbfd_config.n_overlap
    n_block = config.pbfd_config.n_block
    n_frame = (n_time + hop_len - 1) // hop_len
    n_bin = fft_len // 2 + 1

    if n_block * hop_len != rir.shape[1]: 
        raise ValueError(f"n_block * hop_len must equal to rir_len, but got {n_block} * {hop_len} != {rir.shape[1]}.")
    if n_overlap != 2 or fft_len != 2 * hop_len:
        raise ValueError(f"n_overlap must be 2 and fft_len must be 2 times of hop_len, \
                         but got n_overlap={n_overlap}, fft_len={fft_len}, hop_len={hop_len}.")

    target = torch.cat([target, torch.zeros((n_batch, n_frame * hop_len - n_time, n_mic), \
                                           device=device)], dim=1)
    target = target.reshape(n_batch, -1, hop_len, n_mic).permute(0, 2, 1, 3) #[n_batch, hop_len, n_frame, n_mic]
    interf = torch.cat([interf, torch.zeros((n_batch, n_frame * hop_len - n_time, n_mic), 
                                           device=device)], dim=1)
    interf = interf.reshape(n_batch, -1, hop_len, n_mic).permute(0, 2, 1, 3) #[n_batch, hop_len, n_frame, n_mic]
    target_plus_interf = target + interf #[n_batch, hop_len, n_frame, n_mic]
    enhanced = torch.zeros((n_batch, hop_len, n_frame), device=device) #[n_batch, hop_len, n_frame]
    enhanced[:, :, 0] = target[:, :, 0, 0] # initialize the first frame as target

    loudspeaker = torch.zeros(n_batch, hop_len, n_frame, device=device) #[n_batch, hop_len, n_frame]
    loudspeaker_buffer = torch.zeros(n_batch, (n_block + 1) * hop_len, device=device) #[n_batch, rir_len + hop_len]
    fd_loudspeaker_buffer = torch.zeros(n_batch, n_bin, n_block, dtype=torch.complex64, device=device) #[n_batch, n_bin, n_block]
    mic = torch.zeros(n_batch, hop_len, n_frame, n_mic, device=device) #[n_batch, hop_len, n_frame, n_mic]
    rir = rir.reshape(n_batch, -1, hop_len, n_mic).permute(0, 2, 1, 3) #[n_batch, hop_len, n_block, n_mic]
    fd_rir = torch.fft.rfft(rir, fft_len, dim=1)  # [n_batch, n_bin, n_block, n_mic]
    fd_rir = torch.flip(fd_rir, dims=[2]) # flip thr order of blocks for pdfd convolution

    for t in range(n_frame - delay - 1):
        if model_name == 'IdealAFC_DNSF': # IdealAFC_DNSF does not consider acoustic feedback
            mic[:, :, t+1, :] = target_plus_interf[:, :, t+1, :]
        else:
            loudspeaker[:, :, t+delay] = gain * enhanced[:, :,  t]
            loudspeaker[:, :, t+delay] = torch.clamp(loudspeaker[:, :, t+delay], -maxamp, maxamp)

            loudspeaker_buffer[:, :n_block * hop_len] = loudspeaker_buffer[:, hop_len:].clone()
            loudspeaker_buffer[:, n_block * hop_len:] = loudspeaker[:, :, t]
            fd_loudspeaker_buffer[:, :, :n_block-1] = fd_loudspeaker_buffer[:, :, 1:].clone()
            fd_loudspeaker_buffer[:, :, -1] = torch.fft.rfft(loudspeaker_buffer[:, -fft_len:], fft_len, dim=1)
            fd_real_feedback_hat = torch.sum(fd_loudspeaker_buffer.unsqueeze(-1) * fd_rir, dim=2) #[n_batch, n_bin, n_mic]
            real_feedback = torch.fft.irfft(fd_real_feedback_hat, fft_len, dim=1)[:, hop_len:, :] #[n_batch, hop_len, n_mic]

            mic[:, :, t+1, :] = target_plus_interf[:, :, t+1, :] + real_feedback
            
        if config[model_name].infer_setup.require_loudspeaker_data:
            enhanced[:, :, t+1] = model(mic[:, :, t+1, :], loudspeaker_buffer)
        else:
            enhanced[:, :, t+1] = model(mic[:, :, t+1, :])

        output = enhanced.permute(0, 2, 1).reshape(n_batch, -1) #[n_batch, n_time]
    
    return output

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
