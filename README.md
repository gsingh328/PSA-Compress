# Compressing Neural Networks Using Learnable 1-D Non-Linear Functions

Authors: Gaurav Singh and Kia Bazargan

This directory serves as a way to re-create the results from our article - _"Compressing Neural Networks Using Learnable 1-D Non-Linear Functions"_.

If you have used our work, please cite our paper:

```
@article{10.1145/3705926,
author = {Singh, Gaurav and Bazargan, Kia},
title = {Compressing Neural Networks Using Learnable 1-D Non-Linear Functions},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1936-7406},
url = {https://doi.org/10.1145/3705926},
doi = {10.1145/3705926},
abstract = {As deep learning models grow in size to achieve state-of-the-art accuracy, there is a pressing need for compact models. To address this challenge, we introduce a novel operation called Personal Self-Attention (PSA). It is specifically designed to learn non-linear 1-D functions, enhancing existing spline-based methods while remaining compatible with gradient backpropagation. By integrating these non-linear functions with linear transformations, we can achieve the accuracy of larger models but with significantly smaller hidden dimensions, which is crucial for FPGA implementations.We evaluate PSA by implementing it in a Multi-Layer Perceptron (MLP)-based vision model, ResMLP, and testing it on the CIFAR-10 classification task. MLP is gaining increasing popularity due to its widespread use in large-language models. Our results confirm that PSA achieves equivalent accuracy with a 2x smaller hidden size compared to conventional MLPs. Furthermore, by quantizing our non-linear function into a simple lookup table, we reduce the number of operations required by 45\%-28\%, which offers significant benefits for hardware accelerators. To showcase this, we design an end-to-end unrolled streaming accelerator for ResMLP, demonstrating that our compressed model maintains an 88\% accuracy while reducing LUT + DSP resource requirements by 25\%, and doubling throughput to 32kFPS. Additionally, we implement a fixed-size SIMD accelerator for the same compressed model that achieves a 62.1\% improvement in throughput while only consuming 3.5\% extra LUTs.},
note = {Just Accepted},
journal = {ACM Trans. Reconfigurable Technol. Syst.},
month = dec,
keywords = {FPGA accelerator, neural networks, model compression, non-linear functions, paramaterized activation function, lookup table}
}
```

## Requirements

Our tests were run with the following pytorch version. If you are using different version, the exact final accuracy numbers may differ:

```
pytorch=2.3.0
```

Other required libraries are: `torchvision, numpy, matplotlib, tqdm`. Full list of conda environment can be found in [requirements.txt](requirements.txt).

Additionally, the primary dataset we used was CIFAR-10. By default the scripts will not download the CIFAR-10 dataset, and assumes the downloaded location is at `~/icml_data`. You can change these defaults within each `*.sh` scripts by changing the following variables:

```sh
DATASET_DIR="~/icml_data"

# Select the second option if you want to download the dataset
DATASET_DOWNLOAD_FLAG=""
# DATASET_DOWNLOAD_FLAG="--download-dataset"
```

## Software Verification

Download the trained models archive from [here](https://drive.google.com/uc?id=1Z2Bja_5nlB2InSX0KuGOsZse4n6RAqaY&export=download).
This archive has the output folder that includes all the necessary models required to verify the Baseline (ReLU), B-Spline, and our LERP-PSA models accuracy. Ensure the working folder has the following structure and files:

```
├── cifar10_main.py
├── output
│   └── cifar10
│       └── resmlp_layers_4_h_4_embed_96
│           ├── baseline
│           │   ├── h_div_4_model.bin
│           │   ├── h_div_2_model.bin
│           │   ├── h_mul_1_model.bin
│           │   └── h_mul_2_model.bin
│           ├── bspline_65_1e-5
│           │   ├── h_div_4_model.bin
│           │   ├── h_div_2_model.bin
│           │   ├── h_mul_1_model.bin
│           │   └── h_mul_2_model.bin
│           └── lerp_psa_64
│               ├── h_div_4_model.bin
│               ├── h_div_2_model.bin
│               ├── h_mul_1_model.bin
│               └── h_mul_2_model.bin
```

From the working folder, you can run the following scripts to verify the accuracy of each model:

```console
$ ./scripts/run_eval_baseline.sh
$ ./scripts/run_eval_bspline.sh
$ ./scripts/run_eval_lerp_psa.sh
```

| Act | h = 2.0 | h = 1.0 | h = 0.5 | h = 0.25 |
| :---: | :---: | :---: | :---: | :---: |
| Baseline (ReLU) | 87.8 | 86.2 | 83.3 | 78.8 |
| B-Spline (64, 1e-5) | 89.1 | 87.7 | 85.6 | 84.1 |
| LERP-PSA (64) | 89.0 | 88.3 | 85.9 | 84.2 |

To run the GPU kernel runtime statistics (like latency, and memory consumed), you can run:

```console
$ python kernel_time_mes.py lerp_psa
$ python kernel_time_mes.py bspline
```

This tests it with grid size of 64, and hidden expansion ratio (for PSA) of 8. This is also the configuration that was used in the accuracies reported above. To change settings like grid size, expansion size of PSA, and the `save_memory` flag modify the following variables in the `kernel_time_mes.py` script (line ~62):

```python
grid_size = 2**6
exsize = 8
save_memory = True
```

## Hardware Verification

To test the FPGA hardware, please refer to our [FPGA README](fpga/README.md).

## Acknowledgments

This work was supported in part by the Cisco Systems, Inc. under Grant 1085913, and in part by the National Science Foundation under Grant PFI-TT 2016390.
