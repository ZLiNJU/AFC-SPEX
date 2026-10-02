"""Original partitioned KF-AFC simulation and ideal speech replacement."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SE_NN(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        self.L = L

    def forward(self, res, tgt, v, ideal_se=1):
        if ideal_se:
            res_post = tgt
        else:
            mask = tgt.abs() / (v.abs() + 1e-10)
            res_post = res[:, 0].reshape(-1, 1) * mask
        return res_post


class gen_rir:
    def __init__(
        self, audio, fs, room_dim=[10, 10, 10], src=[1, 3, 1],
        mic=[1, 1, 1], rt60=0.5,
    ):
        self.audio = audio
        self.fs = fs
        self.room_dim = room_dim
        self.src = src
        self.mic = mic
        self.rt60 = rt60

    def rir(self):
        import pyroomacoustics as pra

        absorption, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        self.room = pra.ShoeBox(
            self.room_dim,
            materials=pra.Material(absorption),
            fs=self.fs,
            max_order=max_order,
        )
        for source in self.src:
            self.room.add_source(source, signal=self.audio)
        self.room.add_microphone_array(
            pra.MicrophoneArray(np.array([self.mic]).T, self.room.fs)
        )
        self.room.compute_rir()
        return self.room.rir


class KG_Baseline(nn.Module):
    def __init__(self, L=2):
        super().__init__()
        self.L = L

    def forward(self, xt, h_prior, P1, phi_e, phi_f, phi_n, if_cuda):
        alpha = 0.99999
        beta = 0.1
        U = torch.zeros(xt.shape).to(if_cuda)
        kg = torch.zeros(xt.shape, dtype=torch.complex64).to(if_cuda)
        P0 = torch.zeros(xt.shape).to(if_cuda)

        for m in range(xt.shape[1]):
            phi_f[:, m] = beta * phi_f[:, m] + (1 - beta) * h_prior[:, m].abs() ** 2
            phi_n[:, m] = (1 - alpha**2) * phi_f[:, m]
            U[:, m] = xt[:, m].abs() ** 2 * P1[:, m]
            R = U[:, m] + self.L * phi_e + 1e-10
            kg[:, m] = P1[:, m] * xt[:, m].conj() / R
            P0[:, m] = (
                P1[:, m].detach()
                - 1 / self.L * kg[:, m].detach().abs() ** 2 * R.detach()
            )
            P1[:, m] = alpha**2 * P0[:, m] + phi_n[:, m]

        return kg.unsqueeze(2), P1, phi_f, phi_n


class process(nn.Module):
    def __init__(self, L=2, B=4, R=256):
        super().__init__()
        self.L = L
        self.B = B
        self.R = R
        self.kg_baseline = KG_Baseline(L=L)
        self.se_nn = SE_NN(L=L)

    def forward(
        self, tgt_reverb, v, h, amp, if_cuda,
        AFC=1, KF_AFC=1, close_loop=1, ideal_se=1,
    ):
        device = v.device
        n_mic = v.shape[1]
        original_len = v.shape[0]
        n_frames = int(np.ceil(original_len / self.R))
        padding = torch.zeros((self.R * n_frames - original_len, n_mic), device=device)
        v = torch.cat([v, padding], dim=0)
        v_t = v.reshape(-1, self.R, n_mic).transpose(0, 1)

        tgt_reverb = torch.cat([tgt_reverb, padding], dim=0)
        tgt_reverb_t = tgt_reverb.reshape(-1, self.R, n_mic).transpose(0, 1)

        h_t = h.reshape(-1, self.R, n_mic).transpose(0, 1)
        H = torch.fft.rfft(h_t, self.L * self.R, dim=0).flip(dims=[1])
        n_freq = self.L * self.R // 2 + 1

        H_prior = torch.zeros(n_freq, self.B, n_mic, dtype=torch.complex64, device=device)
        H_post = torch.zeros_like(H_prior)

        x_t = torch.zeros(self.R, n_frames, n_mic, device=device)
        x2_t = torch.zeros(self.L * self.R, n_mic, device=device)
        x2 = torch.zeros(n_freq, n_frames, n_mic, dtype=torch.complex64, device=device)
        Y = torch.zeros_like(x2)
        RES = torch.zeros_like(x2)
        res = torch.zeros(self.R, n_frames, n_mic, device=device)
        res_post = torch.zeros_like(res)
        y = torch.zeros_like(res)

        delay = 2
        P_init = torch.zeros(n_freq, self.B, n_mic, device=device)
        P_init[:, 3, :] = 0.8
        P_init[:, 2, :] = 0.4
        P_init[:, 1, :] = 0.2
        P_init[:, 0, :] = 0.1
        res_post[:, 0, :] = tgt_reverb_t[:, 0, 0].reshape(-1, 1)

        for t in range(n_frames - delay - 1):
            if close_loop:
                x_t[:, t + delay, :] = amp * res_post[:, t, :].detach()
            else:
                x_t[:, t + delay, :] = amp * tgt_reverb_t[:, t, :].detach()

            x_t[:, t, :] = torch.clamp(x_t[:, t, :], min=-amp, max=amp)
            x2_t[: (self.L - 1) * self.R, :] = x2_t[self.R :, :]
            x2_t[(self.L - 1) * self.R :, :] = x_t[:, t, :]
            x2[:, t, :] = torch.fft.rfft(x2_t, self.L * self.R, dim=0)

            if t < self.B:
                xt = torch.cat(
                    [
                        torch.zeros(
                            n_freq, self.B - t - 1, n_mic,
                            dtype=torch.complex64, device=device,
                        ),
                        x2[:, : t + 1, :],
                    ],
                    dim=1,
                )
            else:
                xt = x2[:, t - self.B + 1 : t + 1, :]

            if t == 0:
                alpha = 0.99999
                alpha1 = 0.7
                P1 = P_init
                phi_e = torch.zeros(n_freq, n_mic, device=device)
                phi_f = torch.zeros(xt.shape, device=device)
                phi_n = torch.zeros(xt.shape, device=device)

            for i in range(n_mic):
                FB_1 = torch.matmul(xt[..., i].unsqueeze(1), H[..., i].unsqueeze(2)).squeeze()
                fbt = torch.fft.irfft(FB_1, self.L * self.R, dim=0)
                fbt = torch.cat(
                    [
                        torch.zeros((self.L - 1) * self.R, device=device),
                        fbt[(self.L - 1) * self.R :],
                    ],
                    dim=0,
                )
                FB = torch.fft.rfft(fbt, self.L * self.R, dim=0)

                vt = torch.cat(
                    [torch.zeros((self.L - 1) * self.R, device=device), v_t[:, t + 1, i]],
                    dim=0,
                )
                V = torch.fft.rfft(vt, self.L * self.R, dim=0)
                Y[:, t, i] = V + FB

                if AFC:
                    if KF_AFC:
                        H_prior = H_post
                        FB1 = torch.matmul(
                            xt[..., i].unsqueeze(1), H_prior[..., i].unsqueeze(2)
                        ).squeeze()
                        fbt1_t = torch.fft.irfft(FB1, self.L * self.R, dim=0)
                        fbt1 = torch.cat(
                            [
                                torch.zeros((self.L - 1) * self.R, device=device),
                                fbt1_t[(self.L - 1) * self.R :],
                            ],
                            dim=0,
                        )
                        FB_1 = torch.fft.rfft(fbt1, self.L * self.R, dim=0)
                        E = Y[:, t, i] - FB_1

                        phi_e[..., i] = alpha1 * phi_e[..., i] + (1 - alpha1) * E.abs() ** 2
                        kg, P1[..., i], phi_f[..., i], phi_n[..., i] = self.kg_baseline(
                            xt[..., i], H_prior[..., i], P1[..., i],
                            phi_e[..., i], phi_f[..., i], phi_n[..., i], if_cuda,
                        )

                        dh1_t = torch.fft.irfft(
                            torch.matmul(kg, E.unsqueeze(-1).unsqueeze(-1)).squeeze(),
                            self.L * self.R, dim=0,
                        )
                        dh1 = torch.fft.rfft(
                            F.pad(
                                dh1_t[: self.R, :],
                                (0, 0, 0, self.R * (self.L - 1)),
                                mode="constant", value=0,
                            ),
                            dim=0,
                        )
                        H_post[..., i] = H_prior[..., i] + dh1
                        H_post[..., i] = alpha * H_post[..., i]
                    else:
                        H_post[..., i] = H[..., i]

                    FB2 = torch.matmul(
                        xt[..., i].unsqueeze(1), H_post[..., i].unsqueeze(2)
                    ).squeeze()
                    fbt2_t = torch.fft.irfft(FB2, self.L * self.R, dim=0)
                    fbt2 = torch.cat(
                        [
                            torch.zeros((self.L - 1) * self.R, device=device),
                            fbt2_t[(self.L - 1) * self.R :],
                        ],
                        dim=0,
                    )
                    FB_2 = torch.fft.rfft(fbt2, self.L * self.R, dim=0)
                    RES[:, t, i] = Y[:, t, i] - FB_2
                else:
                    RES[:, t, i] = Y[:, t, i]

            res_tf = torch.fft.irfft(RES[:, t, :], self.L * self.R, dim=0)
            res[:, t + 1, :] = res_tf[(self.L - 1) * self.R :, :]
            res[:, t + 1, :] = torch.clamp(res[:, t + 1, :], min=-1, max=1)
            res_post[:, t + 1, :] = self.se_nn(
                res[:, t + 1, :],
                tgt_reverb_t[:, t + 1, 0].reshape(-1, 1),
                v_t[:, t + 1, 0].reshape(-1, 1),
                ideal_se=ideal_se,
            )

            y_tf = torch.fft.irfft(Y[:, t, :], self.L * self.R, dim=0)
            y[:, t + 1, :] = y_tf[(self.L - 1) * self.R :, :]

        y = y.transpose(0, 1).reshape(1, -1, n_mic).squeeze(0)
        res_hat = res.transpose(0, 1).reshape(1, -1, n_mic).squeeze(0)
        return res_hat.cpu(), y.cpu()
