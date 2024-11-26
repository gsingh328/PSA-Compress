# Hardware Verification

This folder contains the two different FPGA accelerators for the ResMLP CIFAR-10 model - fixed SIMD and the unrolled streaming. For both accelerators, there are two versions, one is the baseline that uses the ReLU activation function, while the other utilizes the LERP-PSA and compresses the hidden size by 2x.

The following environment was used:
```
Ubuntu 22.04.4 LTS
Vitis HLS 2023.1
```
When opening the projects with Vitis HLS open the sub-folder, and not the parent folder with Vitis HLS. For example, if opening the `resmlp_strm_baseline` project, open the the `resmlp_strm_baseline/resmlp` folder with Vitis HLS.

By default the HLS project is setup with device of Alveo U50. If you do not have license to this device, please change the device to whatever you are using, but the final synthesis and implementation results may differ.

You can run the simulation and Co-Simulation to verify the hardware accuracy. You may see small differences due to the floating point representation. Final CIFAR-10 accuracy was verified separately using Numpy with purely INT-OPS. The small differences are due to Pytorch being used to save the sample model input and output and it uses floats to simulate quantized model operation.

For Synthesis, the default target period is `5ns` while using the `Vitis Kernal Flow Target`.

For Implementation, `RTL Synthesis, Place & Route` was done using the `default` RTL Synthesis Strategy and `-directive sdx_optimization_effort_high`. Physical optimizations was `none`, and implementation strategy was `default`. Make sure the system has enough system memory (at least 64GB), especially if the target device is the U50.

## Streaming

There are two relevant Vitis HLS projects available:
- resmlp_strm_baseline
- resmlp_strm_psa

For the `resmlp_strm_baseline` project, you can change the hidden size by modifying `#define HIDDEN_N` in `resmlp.h`. There are two valid configurations supported that commented/uncommented based on what hidden size you want to test:

```C++
//#define HIDDEN_N 96
#define HIDDEN_N 192
```
Based on this setting, the appropriate header files will be imported in both the testbench (for testing input/output) and in the accelerator (for setting up the parameters).

For the LERP-PSA implementation, the relevant HLS project is `resmlp_fixed_psa`. There is no hidden size configuration, only a hidden size is 96 currently written.

For measuring throughput, use Co-Sim and measure the min-II of the `write_data` tasks:

| Activation | Hidden Size | Accuracy | II | Throughput |
| -------- | -------- | -------- | ------- | ------- |
| ReLU | 192 | 87.9 | 12352 | 16192 |
| ReLU | 96 | 86.1 | 6240 | 32051 |
| LERP-PSA | 96 | 88.1 | 6240 | 32051 |

The timeline trace can be used separately to verify that this min-II is repeated across the batch of image of size 8, except the first image when the pipeline of the accelerator needs to filled.

## Fixed SIMD
The relevant Vitis HLS projects are:
- resmlp_fixed_baseline
- resmlp_fixed_psa

For the `resmlp_fixed_baseline` project, you can change the hidden size by modifying `#define HIDDEN_N` in `resmlp.h`. There are two valid configurations supported that commented/uncommented based on what hidden size you want to test:

```C++
//#define HIDDEN_N 96
#define HIDDEN_N 192
```
Based on this setting, the appropriate header files will be imported in both the testbench (for testing input/output) and in the accelerator (for setting up the parameters).

For the LERP-PSA implementation, the relevant HLS project is `resmlp_fixed_psa`. There is no hidden size configuration, only a hidden size is 96 currently written.

Both projects are setup to run a single image at a time. So, for measuring throughput, simply use Co-Sim and measure the latency of the entire kernel:

| Activation | Hidden Size | Accuracy | Latency | Throughput |
| -------- | -------- | -------- | ------- | ------- |
| ReLU | 192 | 87.9 |  |  |
| ReLU | 96 | 86.1 |  |  |
| LERP-PSA | 96 | 88.1 |  |  |

