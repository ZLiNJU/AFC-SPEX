"""Generate the original ideal-SE-assisted KF inputs for train, val and test."""

import os
from pathlib import Path

# Edit these paths for the prepared dataset. Input and output may be the same.
DATA_ROOT = Path("/data/ssd4/lize/AFC-SPEX/prep")
OUTPUT_ROOT = DATA_ROOT

# None processes the entire split. The old num=0 setting selected no files.
TRAIN_COUNT = None
VAL_COUNT = 500
TEST_COUNT = None

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from pesq import pesq
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

from dataset import FileDataset
from simulation import process


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"prepared dataset not found: {DATA_ROOT}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    simulator = process(L=2, B=4, R=256).to(device)
    si_sdr = ScaleInvariantSignalDistortionRatio()

    with torch.no_grad():
        for mode, count in (("train", TRAIN_COUNT), ("val", VAL_COUNT), ("test", TEST_COUNT)):
            dataset = FileDataset(
                dir_path=DATA_ROOT, num=count, reverse=True, mode=mode,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
            output_dir = OUTPUT_ROOT / "audio" / mode
            output_dir.mkdir(parents=True, exist_ok=True)

            sisdr_y = 0
            sisdr_res = 0
            pesq_y = 0
            pesq_res = 0
            spr = 0

            for tgt_reverb, noise_interf, rir, file_name in tqdm(loader, desc=mode):
                tgt_reverb = tgt_reverb.squeeze().to(device)
                noise_interf = noise_interf.squeeze().to(device)
                v = tgt_reverb + noise_interf
                h = rir.squeeze().to(device)[:1024, :]

                res_hat, y = simulator(
                    tgt_reverb, v, h, 1, device,
                    AFC=1, KF_AFC=1, close_loop=1, ideal_se=1,
                )

                np.save(output_dir / f"rec_wfb_{file_name[0]}.npy", y.numpy())
                np.save(output_dir / f"afc_res_{file_name[0]}.npy", res_hat.numpy())

                # The simulator pads to a whole 256-sample frame. Exclude that
                # padding when comparing its outputs with the original audio.
                original_len = v.shape[0]
                y_eval = y[:original_len]
                res_eval = res_hat[:original_len]
                for channel in range(3):
                    clean = v[..., channel].cpu()
                    sisdr_y += si_sdr(y_eval[..., channel], clean)
                    sisdr_res += si_sdr(res_eval[..., channel], clean)
                    pesq_y += pesq(16000, clean.numpy(), y_eval[..., channel].numpy(), "nb")
                    pesq_res += pesq(16000, clean.numpy(), res_eval[..., channel].numpy(), "nb")

                fb_signal = y_eval.to(device) - v
                fb_var = np.mean(np.var(fb_signal.cpu().numpy(), axis=0))
                v_var = np.mean(np.var(v.cpu().numpy(), axis=0))
                spr += 10 * np.log10(v_var / (fb_var + 1e-10))

            if not len(loader):
                print(f"{mode}: no input files")
                continue

            scores = {
                "SISDR_y": sisdr_y / len(loader) / 3,
                "SISDR_res": sisdr_res / len(loader) / 3,
                "PESQ_y": pesq_y / len(loader) / 3,
                "PESQ_res": pesq_res / len(loader) / 3,
                "SPR": spr / len(loader),
            }
            for name, value in scores.items():
                print(f"{name}_{mode}: {value:.4f}")
                np.save(OUTPUT_ROOT / f"{name}_{mode}.npy", value)


if __name__ == "__main__":
    main()
