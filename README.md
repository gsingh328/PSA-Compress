# Compressing Neural Networks Using Learnable 1-D Non-Linear Functions

This directory serves as a way to re-create the results from our article - _"Compressing Neural Networks Using Learnable 1-D Non-Linear Functions"_.

The first section covers FP32 training with parent models and distillation of student models. Second section covers quantization to INT8 and accuracy verification using purely INT8 ops inside of numpy to replicate hardware accuracy. Third section covers the hardware implementations on FPGAs.

We provide the final trained models so the software training step (section 1) can be completely skipped. Similarly we also provide the quantized models so the quantization step can be skipped (section 2).

## 0. Requirements

Our tests were run with following version of python, numpy, pytorch and pytorch-cuda. If you are using different versions of pytorch and pytorch-cuda specifically, the exact final accuracy numbers may differ:

```
python=3.11.9
pytorch=2.3.0
pytorch-cuda=12.1
numpy=1.26.4
```

Additionally, the primary dataset we used was CIFAR-10. By default the scripts will not download the CIFAR-10 dataset, and assumes the downloaded location is at `~/icml_data`. You can change these defaults within each `*.sh` scripts listed in this README file by changing the following variables:

```sh
DATASET_DIR="~/icml_data"

# Select the second option if you want to download the dataset
DATASET_DOWNLOAD_FLAG=""
# DATASET_DOWNLOAD_FLAG="--download-dataset"
```

## 1. Software - FP32 Training

Training models from scratch, using absolutely no pre-trained models, is a lengthy process. Hence, we will provide our pre-trained uncompressed ResMLP model that is used to train the distilled compressed ReLU, B-Spline and our LERP-PSA models.

### Pre-Trained Model

Our primary pre-trained model is the uncompressed ResMLP model that has embedding dimension of 96 and hidden size of 384:

| Hidden Size | Accuracy | Download Link |
| -------- | ------- | ------- |
| 384 | 88.9 | [teacher_model.bin]() |

Our training scripts assume this folder structure so please copy the downloaded model as follows:

```
.
├── cifar10_main.py
| .
| .
| .
├── output
│   └── cifar10
│       ├── resmlp_layers_4_h_4_embed_96
│       │   └── teacher_model.bin               <=== PASTE MODEL HERE
```

### Baseline - ReLU

To train the baseline ReLU compressed models, run the script:

```
./scripts/run_compressed_baseline.sh
```

After running the script there should be a new folder `resmlp_layers_4_h_4_embed_96/baseline` folder with the compressed models trained with hidden sizes (192, 96, 48, 24). The accuracy can be verified by checking respective log file and seeing the last accuracy reported.

The primary two models we provide as downloads are at hidden sizes of 192 and 96 since they will be used in quantization and hardware section next:

| Hidden Size | Accuracy | Download Link |
| -------- | ------- | ------- |
| 192 | 87.8 | [h_mul_2_model.bin]() |
| 96 | 86.2 | [h_mul_1_model.bin]() |

If downloading the pre-trained models, please paste them here to be compatible with hardware scripts in next two sections:
```
.
├── cifar10_main.py
| .
| .
| .
├── output
│   └── cifar10
│       └── resmlp_layers_4_h_4_embed_96
│           ├── baseline
│           │   ├── h_mul_1_model.bin           <=== PASTE MODELS HERE
│           │   └── h_mul_2_model.bin
```

### B-Spline

To train the baseline B-Spline compressed models, run the script:

```
./scripts/run_compressed_bspline.sh
```

***By default the B-Spline model is configured with grid size of 65, and $\lambda=1e-5$ for the TV2 loss.***

To change $\lambda$ you will need to modify the following line in the `run_compressed_bspline.sh` file:
```sh
BSPLINE_LAMBDA="1e-5"
```

To change the grid size you will need to modify the python file `models/bspline_resmlp.py`:
```python
_GRID_SIZE=65
```

After running the script there should be a new folder `resmlp_layers_4_h_4_embed_96/bspline_65_1e-5` with the compressed models trained with hidden size (192, 96, 48, 24). The accuracy can be verified by checking respective log file and seeing the last accuracy reported. These should match the accuracies reported in the paper with grid size of 64.

_NOTE: B-Spline requires and odd number in the grid size, but in the paper the grid sizes were subtracted by 1 for clarity._

### LERP-PSA

To train our LERP-PSA models, run the script:

```
./scripts/run_compressed_lerp_psa.sh
```

***By default the LERP-PSA is configured with grid size of 64.***

To change the grid size you will need to modify the python file `models/psa_resmlp.py`:
```python
_GRID_SIZE=64
```

This script should create a new `lerp_psa_64` folder inside the `resmlp_layers_4_h_4_embed_96` folder, with the compressed models trained at hidden size in (192, 96, 48, 24). The accuracy can be verified by checking respective log file and seeing the last accuracy reported.

The primary model we care about is at hidden size of 96, since that is going to be used for quantization and then hardware implementation. Hence, we are also providing the download link for that model:

| Hidden Size | Accuracy | Download Link |
| -------- | ------- | ------- |
| 96 | 88.3 |  |

If downloading the pre-trained model, please paste it here to be compatible with hardware scripts in next two sections:
```
.
├── cifar10_main.py
| .
| .
| .
├── output
│   └── cifar10
│       └── resmlp_layers_4_h_4_embed_96
│           ├── lerp_psa_64
│           │   └── h_mul_1_model.bin            <=== PASTE MODEL HERE
```

## 2. Software - INT8 Quantization + Verification

This section covers the following steps:
- Converts the FP32 model to INT8 using quantization aware training
- Dumps the INT8 parameters in a dictionary pickle format
- Using numpy re-verify the accuracy using purely Numpy's integer OPS that emulate hardware implementation
- Convert parameters and sample input/output to a C header file to be used in HLS code in Section 3.

As always this section can be entirely optional since the header files are already provided in the HLS projects.

Before starting this section, the following models should be either trained or downloaded:
- Uncompressed teacher model with hidden size of 384
- Compressed Baseline ReLU models with hidden sizes of 192 and 96
- Compressed LERP-PSA model with hidden size of 96


The training scripts will assume this directory structure should look like this:

```
.
├── cifar10_quant.py
| .
| .
| .
├── output
│   └── cifar10
│       └── resmlp_layers_4_h_4_embed_96
│           ├── baseline
│           │   ├── h_mul_1_model.bin
│           │   └── h_mul_2_model.bin
│           ├── lerp_psa_64
│           │   └── h_mul_1_model.bin
│           └── teacher_model.bin
```

## 3. Hardware - FPGA Accelerators

## Acknowledgments

This work was supported in part by the Cisco Systems, Inc. under Grant 1085913, and in part by the National Science Foundation under Grant PFI-TT 2016390.
