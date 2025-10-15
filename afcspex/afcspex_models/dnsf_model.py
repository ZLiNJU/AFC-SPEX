from typing import Literal
import torch
import torch.nn as nn

class DNSF(nn.Module):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.

    When streaming_mode is set to True, remember to call reset_states() between different sequences.
    """
    def __init__(self,
                 n_channels: int,
                 n_lstm_hidden1: int = 256,
                 n_lstm_hidden2: int = 128,
                 streaming_mode: bool = False,
                 bidirectional_1: bool = True,
                 bidirectional_2: bool = False,
                 output_type: Literal['IRM', 'CRM'] = 'CRM',
                 output_activation: Literal['sigmoid', 'tanh', 'linear'] = 'tanh',
                 ):
        """
        Initialize model.

        :param n_channels: number of channel in the input signal
        :param n_lstm_hidden1: number of LSTM units in the first LSTM layer
        :param n_lstm_hidden2: number of LSTM units in the second LSTM layer
        :param streaming_mode: if True, the model will maintain LSTM states between consecutive forward passes
        :param bidirectional: set to True for a bidirectional LSTM
        :param freq_first: process frequency dimension first if freq_first else process time dimension first
        :param output_type: output_type: set to 'IRM' for real-valued ideal ratio mask (IRM) and to 'CRM' for complex IRM
        :param output_activation: the activation function applied to the network output (options: 'sigmoid', 'tanh', 'linear')
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_lstm_hidden1 = n_lstm_hidden1
        self.n_lstm_hidden2 = n_lstm_hidden2
        self.bidirectional_1 = bidirectional_1
        self.bidirectional_2 = bidirectional_2
        self.output_type = output_type
        self.output_activation = output_activation
        self.streaming_mode = streaming_mode 
        self.h_state = None
        self.c_state = None

        lstm_input = 2*n_channels
        self.lstm1 = nn.LSTM(input_size=lstm_input, hidden_size=self.n_lstm_hidden1, bidirectional=bidirectional_1, batch_first=False)
        self.lstm1_out = 2*self.n_lstm_hidden1 if self.bidirectional_1 else self.n_lstm_hidden1
        lstm2_input = self.lstm1_out

        self.lstm2 = nn.LSTM(input_size=lstm2_input, hidden_size=self.n_lstm_hidden2, bidirectional=bidirectional_2, batch_first=False)
        self.lstm2_out = 2*self.n_lstm_hidden2 if self.bidirectional_2 else self.n_lstm_hidden2
        
        if self.output_type == 'IRM':
            self.linear_out_features = 1
        elif self.output_type == 'CRM':
            self.linear_out_features = 2
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
        self.ff = nn.Linear(self.lstm2_out, out_features=self.linear_out_features)

        if self.output_activation == 'sigmoid':
            self.mask_activation = nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.mask_activation = nn.Tanh()
        elif self.output_activation == 'linear':
            self.mask_activation = nn.Identity()
    
    def reset_states(self):

        self.h_state = None
        self.c_state = None

    def forward(self, x: torch.Tensor):
        """
        Implements the forward pass of the model.

        :param x: input with shape [BATCH, CHANNEL, FREQ, TIME]
        :return: the output mask [BATCH, 1 (IRM) or 2 (CRM) , FREQ, TIME]
        """
        n_batch, n_channel, n_freq, n_times = x.shape  # [B, C, F, T]

        if self.streaming_mode:
            num_directions = 2 if self.bidirectional_2 else 1
            expected_dims = (self.lstm2.num_layers * num_directions, n_batch * n_freq, self.lstm2.hidden_size)

            if self.h_state is None or self.h_state.shape != expected_dims:
                device = x.device
                self.h_state = torch.zeros(expected_dims, device=device)
                self.c_state = torch.zeros(expected_dims, device=device)
            hidden_tuple = (self.h_state, self.c_state)
        else:
            hidden_tuple = None

        # wide_band
        # [B, C, F, T] -> [F, B, T, C] -> [F, B*T, C]
        x = x.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)
        x, _ = self.lstm1(x)
        # wide_band -> narrow_band
        # [F, B, T, L1_OUT] -> [T, B, F, L1_OUT] -> [T, B*F, L1_OUT]
        x = x.reshape(n_freq, n_batch, n_times, self.lstm1_out).permute(2,1,0,3).reshape(n_times, n_batch*n_freq, self.lstm1_out)
        x, (h_new, c_new) = self.lstm2(x, hidden_tuple) 

        if self.streaming_mode:
            self.h_state = h_new.detach()
            self.c_state = c_new.detach()

        ## FF
        # [T, B*F, L2_OUT] -> [T, B*F, 1 (IRM) or 2 (CRM)]
        x = self.ff(x)
        # narrow_band -> input shape
        # [T, B, F, 1 (IRM) or 2 (CRM)] -> [B, 1 (IRM) or 2 (CRM), F, T]
        x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)
        x = self.mask_activation(x)
        return x