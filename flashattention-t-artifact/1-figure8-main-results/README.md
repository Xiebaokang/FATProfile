# 1. Introduction

This directory contains the artifacts necessary to reproduce the main results presented in Figure 8 of the FlashAttention-T paper on A100 and H100 machines. AGX Orin is not included due to its ARM-based CPU and jetpack OS release model significantly complicating the environment setup process. This evaluation takes around 1 hour to complete on both A100 and H100 machines.

The subdirs/files under this directory are structured as follows:

```
1-figure8-main-results
├── flashattention-t
│   ├── ampere          <- Contains FA-T ILP implementation for Ampere GPUs and benchmark prog
│   └── hopper          <- Contains FA-T TLP implementation for Hopper GPUs and benchmark prog
├── flashinfer          <- Contains scripts for benchmarking flashinfer attention performance
├── plot                <- Contains script for producing perf plot similar to Figure 8
├── README.md           <- This file
└── triton              <- Contains script for benchmarking triton attention performance
```

The system requirements for conducting this evaluation are as follows:

- System: Modern Linux distros
- GPU: NVIDIA A100 and H100
- CUDA Version: 12.8/12.9
- Python Version: 3.11/3.12

In a practical setup, the evaluations might need to be conducted on two separate machines (e.g. machine A with A100, and machine H with H100). Thus, in the following Section 2 and 3, we present complete guides on the two machines separately. After you have collected benchmark csv reports following Section 2 and 3, you can proceed to the Section 4 to produce a perf plot similar to Figure 8 in our paper.

## 1.1 FlashAttention-T Implementation Details

FlashAttention-T (FA-T) is implemented with approximately 4K lines of CUDA/C++ code on top of the original FlashAttention-2/3 codebase. The implementation is organized under the `1-figure8-main-results/flashattention-t/ampere` (ILP FlashAttention-T) and `1-figure8-main-results/flashattention-t/hopper` (TLP FlashAttention-T) directories for A100 (Ampere) and H100 (Hopper) GPUs respectively.

The core ILP FA-T implementation for Ampere GPUs includes (but not limited to) the following files under `1-figure8-main-results/flashattention-t/ampere/csrc/flash_attn/src/` directory:

- `custom_meta.cuh`: Defines several compile-time meta template structs for determining/analyzing register layouts for several key data storages for such as the 16-row max surrogate introduced in Section 4 of the paper.
- `softmax_add.cuh`: Implements several methods for conducting attention score matrix S row summation, including the tensorized approach introduced in Section 3.2 of the paper.
- `softmax_mma_acc_o_rescale.cuh`: Implements several methods for conducting the attention output rescaling, including the tensorized approach introduced in Section 3.1 of the paper.
- `softmax_mma_acc_s_rescale.h`: Implements several methods for conducting the attention score matrix S rescaling, including the tensorized approach introduced in Section 3.1 of the paper.
- `softmax_mma.h`: Implements the 16-row max surrogate computation (Section 4 of the paper), tensor MMA instruction repurposing operand B fragment generation (Section 3 of the paper), and the ILP softmax scheduling (Section 5 of the paper).

The core TLP FA-T implementation for Hopper GPUs includes (but not limited to) the following files under `1-figure8-main-results/flashattention-t/hopper/` directory:

- `custom_meta.cuh`: Defines several compile-time meta template structs for determining/analyzing shared memory layouts for the tensor WGMMA instruction repurposing operand B fragments introduced in Section 3 of the paper.
- `softmax_wgmma_reduce.cuh`: Implements core softmax computation with WGMMA tensorized attention score matrix S row summation computation introduced in Section 3 of the paper.
- `mainloop_fwd_sm90_tma_gmma_ws.hpp`: Modified on top of the original FlashAttention-3 attention mainloop to integrate the proposed TLP scheduling introduced in Section 5 of the paper.

# 2. Reproducing Main Results (Figure 8) on A100

This evaluation aims to compare the FP16-FP32 attention throughput of the following fused attention implementations on the A100 GPU:
- FlashAttention-T (with ILP scheduling)
- FlashAttention-2
- Triton
- FlashInfer

## 2.1 FlashAttention-T & FlashAttention-2 Benchmarking
### 2.1.1 Virtual Environment Setup

Unlike the original FlashAttention-2 work (`flashattention` repo) where the attention implemenetations are compiled/wrapped into a python library and then evaluated via python scripts, in this prototype FlashAttention-T implementation, we directly compile the attention implementations (FlashAttention-T and also FlashAttention-2) into standalone CUDA/C++ binaries and evaluate them via command line interface (CLI). This allows us to have more control over the compilation and also significantly reduce the evaluation time. But, we still need to set up a python virtual environment to install the required dependencies for building the binaries. To set up the virtual environment, please follow the steps below:

```sh
# 1. make sure to create the venv under the correct directory
cd flashattention-t/ampere/

# 2. create venv
# NOTE1: please make sure to use the correct python version (3.11/3.12)
# NOTE2: please make sure the venv is named exactly as ".venv" as our CMakeLists.txt hardcodes this path
python3 -m venv .venv

# 3. activate venv
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 4. install dependencies
# 4.1 pytorch cu129 (or pytorch cu128, both are fine)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
# 4.2 building tools
pip3 install ninja cmake

# 5. deactivate and re-activate the venv once to make sure that the building tools are correctly overriden by the venv
deactivate && source ./.venv/bin/activate
# check these point to the binaries inside the venv
which ninja
which cmake

```

### 2.1.2 Building the Benchmark Executables

After the virtual environment is set up, we can proceed to build the benchmark executables for FlashAttention-T and FlashAttention-2. Please follow the steps below:

```sh

# pre-requisite: make sure to execute the following with the venv activated

# 1. navigate to the benchmark directory
cd flashattention-t/ampere/cxx-tests/fwd_bench
mkdir build
cd build

# 2. configure the build system
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES=80

# 3. build the benchmark executables. This will take for ~3 minutes
# NOTE: you could safely ignore all warnings emitted (especially the ones rooting from the cute library, e.g., warning #20011-D)
ninja
```

As the above building process finishes, two executables are produced under the build directory:
- `fp16-fwd-bench`: For benchmarking the FlashAttention-T implementation
- `fp16-fwd-bench-orig`: For benchmarking the original FlashAttention-2 implementation

### 2.1.3 Running the Benchmarks

Run the benchmark executables with arguments so that the benchmark report csvs are generated. Please follow the steps below:

```sh
# execute the executables for benchmark report csvs
# 1. FlashAttention-T
./fp16-fwd-bench ./data_ours_a100.csv
# 2. FlashAttention-2
./fp16-fwd-bench-orig ./data_fa2_a100.csv

# 3. deactivate venv after benchmarks
deactivate
```

After the above commands finish, two csv files are generated under the build directory:

- `data_ours_a100.csv`: The benchmark report for FlashAttention-T on A100, with FP16-FP32 precision under various attention configurations.
- `data_fa2_a100.csv`: The benchmark report for FlashAttention-3 on A100, with FP16-FP32 precision under various attention configurations.

You could copy the csv files into `flashattention-t-artifact/1-figure8-main-results/plot/` directory for plotting later:

```sh
cp ./data_ours_a100.csv ../../../../../plot/
cp ./data_fa2_a100.csv ../../../../../plot/
```


## 2.2 Triton Benchmarking
### 2.2.1 Virtual Environment Setup


```sh
cd triton/
# 1. create a new venv
python3 -m venv .venv

# 2. activate venv under "triton" directory
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 3. install triton 3.3.1 and other packages
pip3 install pytest matplotlib pandas
# Note: triton 3.3.1 requires this specific torch version
pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install triton==3.3.1
```

Note: please be sure to create a new venv for benchmarking triton following the above steps to avoid conflicts between the packages.

### 2.2.2 Running Triton Benchmark

```sh
# pre-requisite: make sure to execute the following with the venv activated

python3 06-fused-attention.py data_triton_a100.csv

# deactivate venv after benchmark
deactivate
```

After the above command finishes, a csv report is generated under the `triton` directory:

- `data_triton_a100.csv`: The benchmark report for Triton fused attention on A100, with FP16-FP32 precision under various attention configurations.

You could copy the csv file into `flashattention-t-artifact/1-figure8-main-results/plot/` directory for plotting later:

```sh
cp ./data_triton_a100.csv ../plot/
```


## 2.3 FlashInfer Benchmarking
### 2.3.1 Virtual Environment Setup

```sh
cd flashinfer/
# 1. create a new venv
python3 -m venv .venv

# 2. activate venv under "flashinfer" directory
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 3. install flashinfer 0.2.7-post1
pip3 install flashinfer-python==0.2.7-post1
```

Note: please be sure to create a new venv for benchmarking flashinfer following the above steps to avoid conflicts between the packages.


### 2.3.2 Running FlashInfer Benchmark

```sh
# pre-requisite: make sure to execute the following with the venv activated

python3 ampere_attention_bench.py data_flashinfer_a100.csv

# deactivate venv after benchmark
deactivate
```

Note: flashinfer uses JIT kernel compilation, which can be quite slow for the first run. Therefore, the above command might seems to be stuck for a while. Please be patient and wait for it to finish.

After the above command finishes, a csv report is generated under the `flashinfer` directory:

- `data_flashinfer_a100.csv`: The benchmark report for FlashInfer on A100, with FP16-FP32 precision under various attention configurations

You could copy the csv file into `flashattention-t-artifact/1-figure8-main-results/plot/` directory for plotting later:

```sh
cp ./data_flashinfer_a100.csv ../plot/
```

## 2.4 Finishing Remarks

If you have followed and successfully finished all benchmarks above, `1-figure8-main-results/plot` should now contain:

- `data_ours_a100.csv`
- `data_fa2_a100.csv`
- `data_triton_a100.csv`
- `data_flashinfer_a100.csv`

Please save them (presumably to your local machine via a simple `cat` and copy-paste) as the later plotting script **requires all csv reports (totalling 8 csv reports) for both A100 and H100** to work.


# 3. Reproducing Main Results (Figure 8) on H100

This evaluation aims to compare the FP8-FP32 attention throughput of the following fused attention implementations on the H100 GPU:
- FlashAttention-T (with TLP scheduling)
- FlashAttention-3
- Triton
- FlashInfer

## 3.1 FlashAttention-T & FlashAttention-3 Benchmarking
### 3.1.1 Virtual Environment Setup

Unlike the original FlashAttention-3 work (`flashattention/hopper` repo) where the attention implemenetations are compiled/wrapped into a python library and then evaluated via python scripts, in this prototype FlashAttention-T implementation, we directly compile the attention implementations (FlashAttention-T and also FlashAttention-3) into standalone CUDA/C++ binaries and evaluate them via command line interface (CLI). This allows us to have more control over the compilation and also significantly reduce the evaluation time. But, we still need to set up a python virtual environment to install the required dependencies for building the binaries. To set up the virtual environment, please follow the steps below:

```sh
# 1. make sure to create the venv under the correct directory
cd flashattention-t/hopper/

# 2. create venv
# NOTE1: please make sure to use the correct python version (3.11/3.12)
# NOTE2: please make sure the venv is named exactly as ".venv" as our CMakeLists.txt hardcodes this path
python3 -m venv .venv

# 3. activate venv
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 4. install dependencies
# 4.1 pytorch cu129 (or pytorch cu128, both are fine)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
# 4.2 building tools
pip3 install ninja cmake

# 5. deactivate and re-activate the venv once to make sure that the building tools are correctly overriden by the venv
deactivate && source ./.venv/bin/activate
# check these point to the binaries inside the venv
which ninja
which cmake
```


### 3.1.2 Building the Benchmark Executables

After the virtual environment is set up, we can proceed to build the benchmark executables for FlashAttention-T and FlashAttention-3. Please follow the steps below:

```sh

# pre-requisite: make sure to execute the following with the venv activated

# 1. navigate to the benchmark directory
cd flashattention-t/hopper/cxx-tests/fwd_bench
mkdir build
cd build

# 2. configure the build system
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES=90a

# 3. build the benchmark executables. This will take for ~3 minutes
ninja
```

As the above building process finishes, two executables are produced under the build directory:
- `fp8-fwd-bench`: For benchmarking the FlashAttention-T implementation
- `fp8-fwd-bench-orig`: For benchmarking the original FlashAttention-3 implementation

Note: Another binary target `libfa3_prepare_scheduler.so` containing necessary scheduler functions for FlashAttention-3 is also built. This dynlib will be dynamically linked by both benchmark executables and you could safely ignore it.

### 3.1.3 Running the Benchmarks

Run the benchmark executables with arguments so that the benchmark report csvs are generated. Please follow the steps below:

```sh
# execute the executables for benchmark report csvs
# 1. FlashAttention-T
./fp8-fwd-bench ./data_ours_h100.csv
# 2. FlashAttention-3
./fp8-fwd-bench-orig ./data_fa3_h100.csv

# 3. deactivate venv after benchmarks
deactivate
```

After the above commands finish, two csv files are generated under the build directory:

- `data_ours_h100.csv`: The benchmark report for FlashAttention-T on H100, with FP8-FP32 precision under various attention configurations.
- `data_fa3_h100.csv`: The benchmark report for FlashAttention-3 on H100, with FP8-FP32 precision under various attention configurations.

You could copy the csv files into `flashattention-t-artifact/1-figure8-main-results/plot/` directory for plotting later:

```sh
cp ./data_ours_h100.csv ../../../../../plot/
cp ./data_fa3_h100.csv ../../../../../plot/
```

## 3.2 Triton Benchmarking
### 3.2.1 Virtual Environment Setup


```sh
cd triton/
# 1. create a new venv
python3 -m venv .venv

# 2. activate venv under "triton" directory
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 3. install triton 3.3.1 and other packages
pip3 install pytest matplotlib pandas
# Note: triton 3.3.1 requires this specific torch version
pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install triton==3.3.1
```

Note: please be sure to create a new venv for benchmarking triton following the above steps to avoid conflicts between the packages.

### 3.2.2 Running Triton Benchmark

```sh
# pre-requisite: make sure to execute the following with the venv activated

python3 06-fused-attention.py data_triton_h100.csv --fp8

# deactivate venv after benchmark
deactivate
```

After the above command finishes, a csv report is generated under the `triton` directory:

- `data_triton_h100.csv`: The benchmark report for Triton fused attention on H100, with FP8-FP32 precision under various attention configurations.

You could copy the csv file into `flashattention-t-artifact/1-figure8-main-results/plot/` directory for plotting later:

```sh
cp ./data_triton_h100.csv ../plot/
```


## 3.3 FlashInfer Benchmarking
### 3.3.1 Virtual Environment Setup

```sh
cd flashinfer/
# 1. create a new venv
python3 -m venv .venv

# 2. activate venv under "flashinfer" directory
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 3. install flashinfer 0.2.7-post1
pip3 install flashinfer-python==0.2.7-post1
```

Note: please be sure to create a new venv for benchmarking flashinfer following the above steps to avoid conflicts between the packages.


### 3.3.2 Running FlashInfer Benchmark

```sh
# pre-requisite: make sure to execute the following with the venv activated

python3 hopper_attention_bench.py data_flashinfer_h100.csv

# deactivate venv after benchmark
deactivate
```

Note: flashinfer uses JIT kernel compilation, which can be quite slow for the first run. Therefore, the above command might seems to be stuck for a while. Please be patient and wait for it to finish.

After the above command finishes, a csv report is generated under the `flashinfer` directory:

- `data_flashinfer_h100.csv`: The benchmark report for FlashInfer on H100, with FP8-FP32 precision under various attention configurations

You could copy the csv file into `flashattention-t-artifact/1-figure8-main-results/plot/` directory for plotting later:

```sh
cp ./data_flashinfer_h100.csv ../plot/
```

## 3.4 Finishing Remarks

If you have followed and successfully finished all benchmarks above, `1-figure8-main-results/plot` should now contain:

- `data_ours_h100.csv`
- `data_fa3_h100.csv`
- `data_triton_h100.csv`
- `data_flashinfer_h100.csv`

Please save them (presumably to your local machine via a simple `cat` and copy-paste) as the later plotting script **requires all csv reports (totalling 8 csv reports) for both A100 and H100** to work.

# 4. Plotting the Results

After obtaining all the necessary benchmark report csv files from the previous steps (presumably saved on your local machine), please place them under the `flashattention-t-artifact/1-figure8-main-results/plot/` directory. The required csv files are:

- `data_ours_a100.csv`
- `data_fa2_a100.csv`
- `data_triton_a100.csv`
- `data_flashinfer_a100.csv`

- `data_ours_h100.csv`
- `data_fa3_h100.csv`
- `data_triton_h100.csv`
- `data_flashinfer_h100.csv`

Now you can proceed to plot the results using the provided plotting script. Please follow the steps below:

```sh
cd flashattention-t-artifact/1-figure8-main-results/plot/
# 1. create a venv for plotting
python3 -m venv .venv
# NOTE: please make sure you have deactivated any other venv before activating this one
source ./.venv/bin/activate

# 2. install required packages
pip3 install matplotlib

# 3. run the plotting script
python3 ./plot.py
```

This will generate a plot pdf named `fig-attention-throughput.pdf` under the `plot` directory, which contains the reproduced main results presented in Figure 8 of the FlashAttention-T paper on both A100 and H100 machines.