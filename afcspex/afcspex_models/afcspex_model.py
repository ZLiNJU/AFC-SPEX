from typing import Literal
import torch
import torch.nn as nn
from .kalman_filter_model import PBFDKalmanFilter
from afcspex_models.dnsf_model import DNSF  
from afcspex_models.kalman_filter_model import PBFDKalmanFilter
from afcspex_models.utils import get_complex_masks_from_stacked
from afcspex_models.utils import PartitionedFFT

class AFC_SPEX(nn.Module):
    '''
    Ideal AFC + DNSF for Acoustic Feedback Cancellation and Noise Suppression
    
    Infact it is a speaker extraction model

    When streaming_mode is set to True, remember to call reset_states() between different sequences.
    '''
    def __init__(self,
                 n_channels: int = 3,
                 n_block: int = 4,
                 state_factor: float = 0.99999,
                 alpha_e: float = 0.7,
                 beta_f: float = 0.1,
                 kalman_filter_len: int = 1024,
                 fft_len: int = 512,
                 hop_len: int = 256,
                 n_lstm_hidden1: int = 256,
                 n_lstm_hidden2: int = 128,
                 streaming_mode: bool = False,
                 bidirectional_1: bool = True,
                 bidirectional_2: bool = False,
                 output_type: Literal['IRM', 'CRM'] = 'CRM',
                 output_activation: Literal['sigmoid', 'tanh', 'linear'] = 'tanh',
                 ):
        super().__init__()
        self.dnsf = DNSF(n_channels=n_channels * 2,
                         n_lstm_hidden1=n_lstm_hidden1,
                         n_lstm_hidden2=n_lstm_hidden2,
                         streaming_mode=streaming_mode,
                         bidirectional_1=bidirectional_1,
                         bidirectional_2=bidirectional_2,
                         output_type=output_type,
                         output_activation=output_activation,
                         )
        self.pbfd_kf = PBFDKalmanFilter(n_channels=n_channels,
                                        n_block=n_block,
                                        state_factor=state_factor,
                                        alpha_e=alpha_e,
                                        beta_f=beta_f,
                                        kalman_filter_len=kalman_filter_len,
                                        fft_len=fft_len,
                                        hop_len=hop_len,
                                        output_type='FrequencyDomain',)
        self.streaming_mode = streaming_mode
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.ft = PartitionedFFT(fft_len=self.fft_len, hop_len=self.hop_len)
        self.ift = PartitionedFFT(fft_len=self.fft_len, hop_len=self.hop_len) 

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.streaming_mode:
                raise ValueError(
                    "For training, streaming_mode must be set to False."
                )
        return self
    
    def reset_states(self):
        self.dnsf.reset_states()
        self.pbfd_kf.reset_states()

    def _offline_forward(self, mic_stack_kf_res: torch.Tensor):
        '''
        Offline processing of the input signal mic

        :param mic_stack_kf_res: input signal [n_batch, n_time, n_mic * 2]
        :return: enhanced signal [n_batch, time], estimated noise signal [n_batch, time], 
                 fd_enhanced_reshape[n_batch, n_bin, n_frame], fd_est_noise_reshape[n_batch, n_bin, n_frame]
        '''
        n_mic = mic_stack_kf_res.shape[2] // 2
        fd_mic_stack_kf_res = self.ft(td_signal=mic_stack_kf_res, fd_signal=None)  # [n_batch, n_time, n_mic * 2] -> [n_batch, n_bin, n_frame, n_mic * 2]
        fd_mic = fd_mic_stack_kf_res[:, :, :, :n_mic].permute(0,3,1,2)
        fd_kf_res = fd_mic_stack_kf_res[:, :, :, n_mic:].permute(0,3,1,2)  # [n_batch, n_mic, n_bin, n_frame]
        fd_mic_stack = torch.concat((torch.real(fd_mic), torch.imag(fd_mic)), dim=1) # [n_batch, n_mic * 2, n_bin, n_frame]
        fd_kf_res_stack = torch.concat((torch.real(fd_kf_res), torch.imag(fd_kf_res)), dim=1) # [n_batch, n_mic * 2, n_bin, n_frame]
        stacked_target_mask = self.dnsf(torch.concat([fd_mic_stack, fd_kf_res_stack], dim=1)) # [n_batch, 2, n_bin, n_frame]
        target_mask, noise_mask = get_complex_masks_from_stacked(stacked_target_mask) # [n_batch, n_bin, n_frame]
        fd_enhanced = fd_mic[:, 0, ...] * target_mask # [n_batch, n_bin, n_frame]
        fd_noise = fd_mic[:, 0, ...] * noise_mask  # [n_batch, n_bin, n_frame]  
        fd_enhanced_reshape = fd_enhanced[:, :, :, None] # [n_batch, n_bin, n_frame, n_mic=1]
        fd_est_noise_reshape = fd_noise[:, :, :, None] # [n_batch, n_bin, n_frame, n_mic=1]
        enhanced = self.ift(td_signal=None, fd_signal=fd_enhanced_reshape) # [n_batch, n_time, n_mic=1]
        est_noise = self.ift(td_signal=None, fd_signal=fd_est_noise_reshape) # [n_batch, n_time, n_mic=1]

        return enhanced.squeeze(-1), est_noise.squeeze(-1), fd_enhanced_reshape.squeeze(-1), fd_est_noise_reshape.squeeze(-1)

    def _streaming_forward(self, mic: torch.Tensor, loudspeaker_buffer: torch.Tensor):
        '''
        Streaming processing of the input signal x

        :param mic: input signal [n_batch, hop_len, n_mic], (one frame)
        :param loudspeaker_buffer: loudspeaker buffer [n_batch, rir_len + hop_len]
        :return: 'enhanced' [n_batch, hop_len] (one frame)
        '''
        device = mic.device
        n_mic = mic.shape[2]
        n_batch = mic.shape[0]
        fd_kf_res = self.pbfd_kf(mic, loudspeaker_buffer)  # [n_batch, n_bin, n_mic]
        mic_zero_pad = torch.cat([torch.zeros([n_batch, self.fft_len - self.hop_len, n_mic], device=device), mic], dim=1)  # [n_batch, fft_len, n_mic]
        fd_mic = torch.fft.rfft(mic_zero_pad, n = self.fft_len, dim=1)  # [n_batch, fft_len, n_mic] -> [n_batch, n_bin, n_mic]
        fd_mic = fd_mic.permute(0, 2, 1)[:, :, :, None] # [n_batch, n_mic, n_bin, n_frame=1]
        fd_mic_stack = torch.concat((torch.real(fd_mic), torch.imag(fd_mic)), dim=1) # [n_batch, 2 * n_mic, n_bin, n_frame=1]
        fd_kf_res = fd_kf_res.permute(0, 2, 1)[:, :, :, None] # [n_batch, n_mic, n_bin, n_frame=1]
        fd_kf_res_stack = torch.concat((torch.real(fd_kf_res), torch.imag(fd_kf_res)), dim=1) # [n_batch, 2 * n_mic, n_bin, n_frame=1]
        stacked_target_mask = self.dnsf(torch.concat([fd_mic_stack, fd_kf_res_stack], dim=1)) # [n_batch, 2, n_bin, n_frame=1]
        target_mask, noise_mask = get_complex_masks_from_stacked(stacked_target_mask) # [n_batch, n_bin, n_frame=1]
        fd_enhanced = fd_mic[:, 0, ...] * target_mask # [n_batch, n_bin, n_frame=1]
        fd_enhanced_reshape = fd_enhanced[:, :, :, None] # [n_batch, n_bin, n_frame=1, n_mic=1]
        enhanced = self.ift(td_signal=None, fd_signal=fd_enhanced_reshape) # [n_batch, hop_len, n_mic=1]
        
        return enhanced.squeeze(-1)
    
    def prepare_train_data(self, target, interf, mic_rec_wfb, afc_res):
        
        noise = mic_rec_wfb - target
        noisy = torch.cat([mic_rec_wfb, afc_res], dim=2)

        return noise, noisy
    
    def forward(self, x, *args):
        if self.streaming_mode:
            return self._streaming_forward(x, *args)
        else:
            return self._offline_forward(x, *args)