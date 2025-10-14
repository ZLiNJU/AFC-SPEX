# AFC-SPEX

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

- Source code for the AFC-SPEX project.
- Dataset and test samples are available at: https://box.nju.edu.cn/d/cd07cf71914c4edfa128/ (Password: `afcspex1014`)

## About The Project

- This repository contains the implementation of the AFC-SPEX algorithm, which is designed for combining multichannel adaptive feedback cancellation with speaker extraction.
- Codes are being rebuilt: the baseline `ideal_afc_dnsf` and the proposed `afc_spex` are both available. More baselines will be added soon; they should be straightforward to implement within the current framework.
- These codes are developed by H.C. Guo and Z. Li.

## Getting Started

- Use the `environment.yml` file to create the conda environment.
- Download the dataset from https://box.nju.edu.cn/d/cd07cf71914c4edfa128/ (Password: `afcspex1014`), dataset name is `prep.zip`.
- `train.py` is the main script for training the model (distributed training supported).
- `infer.py` is the main script for inference; a closed-loop simulation is implemented there.
- Run `evaluate.py` to evaluate the results.
- Configuration files are in the `configs/` folder.

## Acknowledgements

- The architecture is based on the excellent [SEtrain](https://github.com/Xiaobin-Rong/SEtrain) repository.
