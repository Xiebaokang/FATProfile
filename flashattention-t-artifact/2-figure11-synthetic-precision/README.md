# 1. Introduction

This directory contains the artifacts necessary to reproduce the synthetic precision results shown in Figure 11 of the FlashAttention-T paper with A100 GPU. This evaluation takes around 1.5 hours to complete.

The subdirs/files under this directory are structured as follows:


```
2-figure11-synthetic-precision
├── flash-attention-t   <- ILP FA-T source tree for replacing the bundled FA-2 in pytorch
├── make_fat_pytorch.py <- Helper script to create modified PyTorch with FA-T integration
├── plot                <- Plotting script for synthetic precision results
├── prec_bench          <- Synthetic precision benchmark code
├── README.md           <- This file
└── setup_venvs.py      <- Helper script to setup virtual environments

```

The system requirements for conducting this evaluation are as follows:

- System: Modern Linux distros
- GPU: NVIDIA A100
- CUDA Version: 12.8/12.9
- Python Version: 3.11/3.12

# 2. Virtual Environment Setup

In this evaluation, we will first create a modified PyTorch v2.7.0 codebase `pytorch` integrated with FlashAttention-T implementations, under the same directory where this `README.md` file resides. The bundled FlashAttention-2 in this codebase will be replaced with FlashAttention-T. Then, we setup three different virtual environments (venvs) to test different attention implementations:

- `.venv-orig-torch`: This venv is for running the reference pytorch FP64 attention kernel, running the original FlashAttention-2 FP16 attention kernel, conducting the attention precision comparison, and also for plotting the results.
- `.venv-fat-torch`: This venv is for running the ILP FlashAttention-T FP16 attention kernel.
- `.venv-fatmax16-torch`: This venv is for running the "FA2+Max16" ablation variant introduced in Section 6.2 of the paper, which is essentially a baseline FlashAttention-2 implementation with the 16-row surrogate maximum integrated.

Due to the complexity of setting up the above venvs, we offer two helper scripts `make_fat_pytorch` and `setup_venvs.py` to automate the setup process. Please follow the steps below to set up the environments.

## 2.1 Creating the Modified PyTorch with FlashAttention-T Integration

```sh
# navigate to the this directory
cd flashattention-t-artifact/2-figure11-synthetic-precision
python3 ./make_fat_pytorch.py
```

This script will clone the PyTorch v2.7.0 repository, integrate the FlashAttention-T implementation, and create a modified PyTorch codebase in the `pytorch` subdir. This can take around 15 minutes to complete so please be patient.

## 2.2 Setting up the Virtual Environments

```sh
python3 ./setup_venvs.py
```

This script will automatically create the three venvs, install the required dependencies, and build/install the modified PyTorch v2.7.0 with FlashAttention-T integration. It may take around an hour to complete the entire setup process so please be patient. After the setup is complete, you should see the three venvs created under `flashattention-t-artifact/2-figure11-synthetic-precision`.

# 3. Running the Synthetic Precision Benchmark

After the venvs are set up, you can run the synthetic precision benchmark by executing the following command in the terminal:

```sh
# 1. first manually activate the .venv-orig-torch
source ./.venv-orig-torch/bin/activate

# 2. run the synthetic precision benchmark
cd prec_bench
python3 ./loop_runner.py

# 3. deactivate the venv after the benchmark is complete
deactivate
```

This will run the synthetic precision benchmark for all the attention implementations and generate the result files `loop_sdpa_precision_results.json` under the `prec_bench` directory.

# 4. Plotting the Results

After obtaining the benchmark results, you can plot the results by executing the following command in the terminal:

```sh

# reuse the .venv-orig-torch venv for plotting
source ./.venv-orig-torch/bin/activate
cd plot

# move the result file to the plot directory
mv ../prec_bench/loop_sdpa_precision_results.json .

# run the plot script
python3 ./plot.py

# deactivate the venv after plotting
deactivate
```

This will generate a plot pdf named `fig-standalone-fa2-rmse-standard.pdf` under the `plot` directory, which contains the reproduced synthetic precision results presented in Figure 11 of the FlashAttention-T paper.
