from typing import Literal
import torch
import torch.nn as nn

class PBFDKalmanFilter(nn.Module):
    '''
    Partitioned Block Frequency-Domain Kalman Filter for Acoustic Feedback Cancellation

    Input is a frame of microphone and loudspeaker signals
    Output is the enhanced signal frame (Time Domain or Partitioned Block Frequency Domain)

    Remember to call reset_states() between different sequences.
    '''
    
    def __init__(self,
                 n_channels: int = 1,
                 n_block: int = 4,
                 fft_len: int = 512,
                 hop_len: int = 256,
                 state_factor: float = 0.99999,
                 alpha_e: float = 0.7,
                 beta_f: float = 0.1,
                 kalman_filter_len: int = 1024,
                 output_type: Literal['TimeDomain', 'FrequencyDomain'] = 'TimeDomain',
                 ):
        
        if fft_len != 2 * hop_len:
            raise ValueError("fft_len must be equal to 2 * hop_len")
        if kalman_filter_len / n_block != hop_len:
            raise ValueError('kalman_filter_len must be n_block times of hop_len')
        
        super().__init__()
        self.n_channels = n_channels
        self.n_block = n_block
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.state_factor = state_factor
        self.alpha_e = alpha_e
        self.beta_f = beta_f
        self.kalman_filter_len = kalman_filter_len
        self.output_type = output_type
        self.n_bin = self.fft_len // 2 + 1
        self.reset_states()

    def reset_states(self):
        self.phi_e = None
        self.phi_f = None
        self.phi_n = None
        self.H_prior = None
        self.H_post = None
        self.P1 = None

    def _cal_kalman_gain(self, fd_loudspeaker_buffer):
        '''
        :param fd_loudspeaker_buffer: [n_batch, n_bin, n_block]
        :return: kalman_gain: [n_batch, n_bin, n_block, n_mic]
        '''

        self.phi_f = self.beta_f * self.phi_f + (1 - self.beta_f) * self.H_prior.abs() ** 2
        self.phi_n = (1 - self.state_factor ** 2) * self.phi_f
        Xr = fd_loudspeaker_buffer.abs() ** 2
        U = Xr.unsqueeze(-1) * self.P1
        R = U + 2 * self.phi_e.unsqueeze(2) + 1e-10
        kalman_gain = self.P1 * fd_loudspeaker_buffer.unsqueeze(-1).conj() / R
        P0 = self.P1.detach() - 0.5 * kalman_gain.detach().abs() ** 2 * R.detach()
        self.P1 = self.state_factor ** 2 * P0 + self.phi_n

        return kalman_gain

    def forward(self, mic: torch.Tensor, loudspeaker_buffer: torch.Tensor):
        '''
        Kalman filter processing of the input signal mic

        :param mic: input signal [n_batch, hop_len, n_mic]
        :param loudspeaker_buffer: loudspeaker buffer [n_batch, rir_len + hop_len]
        :return: enhanced signal [n_batch, hop_len, n_mic] or [n_batch, n_bin, n_mic]
        '''
        n_batch, _, n_mic = mic.shape
        device = mic.device
        if n_mic != self.n_channels:
            raise ValueError(f'Input mic has {n_mic} channels, \
                             but model is configured for {self.n_channels} channels')
        if loudspeaker_buffer.shape[1] / self.hop_len != self.n_block + 1:
            raise ValueError('loudspeaker_buffer length must be (n_block+1) times of hop_len')
        
        if self.phi_e is None: # initialize states at the first run
            self.P1 = torch.zeros(n_batch, self.n_bin, self.n_block, self.n_channels).to(device)
            if self.n_block == 4: # a special initialization for 4-block KF
                self.P1[:, :, 3, :] = 0.8
                self.P1[:, :, 2, :] = 0.4
                self.P1[:, :, 1, :] = 0.2
                self.P1[:, :, 0, :] = 0.1
            self.phi_e = torch.zeros(n_batch, self.n_bin, self.n_channels).to(device)
            self.phi_f = torch.zeros(n_batch, self.n_bin, self.n_block, self.n_channels).to(device)
            self.phi_n = torch.zeros(self.phi_f.shape).to(device)
            self.H_prior = torch.zeros(n_batch, self.n_bin, self.n_block, self.n_channels, 
                                       dtype=torch.complex64).to(device)
            self.H_post = torch.zeros(self.H_prior.shape, dtype=torch.complex64).to(device)

        fd_loudspeaker_buffer = torch.stft(loudspeaker_buffer,
                                           n_fft=self.fft_len,
                                           hop_length=self.hop_len,
                                           win_length=self.fft_len,
                                           window=torch.ones(self.fft_len, device=device),
                                           center=False,
                                           return_complex=True) #[n_batch, n_bin, n_block]
        mic_zero_pad = torch.cat([torch.zeros([n_batch, self.fft_len - self.hop_len, self.n_channels], 
                                              device=device), mic], dim=1) #[n_batch, fft_len, n_mic]
        fd_mic = torch.fft.rfft(mic_zero_pad, n=self.fft_len, dim=1) #[n_batch, n_bin, n_mic]

        # Update KalmanFilter 
        self.H_prior = self.H_post
        fd_est_feedback_hat = torch.sum(fd_loudspeaker_buffer.unsqueeze(-1) * self.H_prior, dim=2) #[n_batch, n_bin, n_mic]
        est_feedback = torch.fft.irfft(fd_est_feedback_hat, self.fft_len, dim=1)[:, self.hop_len:, :] #[n_batch, hop_len, n_mic]
        est_feedback_zero_pad = torch.cat([torch.zeros([n_batch, self.fft_len - self.hop_len, self.n_channels], 
                                                        device=device), est_feedback], dim=1) #[n_batch, fft_len, n_mic]
        fd_est_feedback = torch.fft.rfft(est_feedback_zero_pad, self.fft_len, dim=1) #[n_batch, n_bin, n_mic]
        E = fd_mic - fd_est_feedback #[n_batch, n_bin, n_mic]
        self.phi_e = self.alpha_e * self.phi_e + (1 - self.alpha_e) * (E.abs() ** 2) #[n_batch, n_bin, n_mic]
        kalman_gain = self._cal_kalman_gain(fd_loudspeaker_buffer) #[n_batch, n_bin, n_block, n_mic]
        dH = torch.fft.irfft(kalman_gain * E.unsqueeze(2), self.fft_len, dim=1) # [n_batch, fft_len, n_block, n_mic]
        dH[:, self.hop_len:, :, :] = 0
        fd_dH = torch.fft.rfft(dH, n=self.fft_len, dim=1) #[n_batch, n_bin, n_block, n_mic]
        self.H_post = self.H_prior + fd_dH
        self.H_post = self.state_factor * self.H_post

        # Perform feedback cancellation
        fd_est_feedback_hat = torch.sum(fd_loudspeaker_buffer.unsqueeze(-1) * self.H_post, dim=2) #[n_batch, n_bin, n_mic]
        est_feedback = torch.fft.irfft(fd_est_feedback_hat, self.fft_len, dim=1)[:, self.hop_len:, :] #[n_batch, hop_len, n_mic]

        if self.output_type == 'TimeDomain':
            enhanced = mic - est_feedback
            return enhanced
        elif self.output_type == 'FrequencyDomain':
            est_feedback_zero_pad = torch.cat([torch.zeros([n_batch, self.fft_len - self.hop_len, self.n_channels],
                                                            device=device), est_feedback], dim=1) #[n_batch, fft_len, n_mic]
            fd_est_feedback = torch.fft.rfft(est_feedback_zero_pad, self.fft_len, dim=1) #[n_batch, n_bin, n_mic]
            fd_enhanced = fd_mic - fd_est_feedback
            return fd_enhanced


