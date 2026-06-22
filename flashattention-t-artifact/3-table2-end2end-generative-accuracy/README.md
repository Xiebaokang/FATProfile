# 1. Introduction

This directory contains the artifacts necessary to reproduce the end-to-end generative benchmark (MMLU) accuracy results shown in Table 2 of the FlashAttention-T paper with A100 GPU. Note that steps for reproducing the HumanEval benchmark results are not included as evaluating the Pass@10 metric requires very prolonged runtimes (24+ hours). This evaluation takes around 1 hour to complete.

The subdirs/files under this directory are structured as follows:

```
3-table2-end2end-generative-accuracy
├── README.md           <- This file
├── setup_venvs.py      <- Helper script for installing required packages in the venvs
└── hf_token_guide      <- A guide for obtaining HuggingFace model access token
```

The system requirements for conducting this evaluation are as follows:

- System: Modern Linux distros
- GPU: NVIDIA A100
- CUDA Version: 12.8/12.9
- Python Version: 3.11/3.12

# 2. Virtual Environment Setup

We will create a dedicated virtual environment (venv) `.venv-download-models` for downloading the models from HuggingFace, and reuse the two venvs set up in the previous synthetic precision evaluation under the `flashattention-t-artifact/2-figure11-synthetic-precision` directory for running the MMLU benchmark with different attention implementations. Thus, while the downloading can be done in the newly created venv, for conducting the MMLU benchmark, please make sure you have followed the instructions in the [README.md](../2-figure11-synthetic-precision/README.md) file under that directory to set up the venvs. Specifically, the following two venvs will be reused in this evaluation with extra package installations:

- `.venv-orig-torch`: This venv is for running the MMLU benchmark with the original FlashAttention-2 attention kernel.
- `.venv-fat-torch`: This venv is for running the MMLU benchmark with the FlashAttention-T attention kernel.

We provide two helper scripts to automate the downloading & venv setup process:

- `download_models.py`: A helper script that automatically creates the `.venv-download-models` venv, installs the required packages `huggingface_hub` and `hf_transfer`, and downloads the required models from HuggingFace with your obtained access token. Please refer to [Section 3](#3-obtaining-huggingface-model-access-token--downloading-models) of this README for detailed instructions.

- `setup_venvs.py`: A helper script that installs the required packages `huggingface_hub` and `lm-evaluation-harness` in both reused venvs for running the MMLU benchmark. Please follow the steps below to set up the environments.

```sh
# NOTE: make sure no venv is activated in the current terminal
# navigate to the this directory
cd flashattention-t-artifact/3-table2-end2end-generative-accuracy
python3 ./setup_venvs.py
```

This script will automatically install the required dependencies in both venvs. Note that the script will check if the two venvs exist under the `flashattention-t-artifact/2-figure11-synthetic-precision` directory, and will exit with an error message if they do not, so please make sure you have set up the venvs as instructed in the previous evaluation.

# 3. Obtaining HuggingFace Model Access Token & Downloading Models

To run the end-to-end generative accuracy benchmarks, you will need to download pre-trained language models from HuggingFace. Specifically, the following models are used in the MMLU benchmark, as introduced in Section 7.5 of the FlashAttention-T paper:

- Llama2 13B ([`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf))
- Mistral Nemo (`[mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)`)
- Qwen3 14B (`[Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)`)

Out of these models, the Llama2 13B model requires you to request access before downloading.

## 3.1 Obtaining HuggingFace Model Access Token

Please refer to the instructions in the [hf_token_guide/README.md](./hf_token_guide/README.md) file to obtain a HuggingFace access token and request access to the Llama2 13B model. In the following sections, we use the placeholder `$hf_YOURTOKEN` to represent the HuggingFace access token you obtained by following the instructions in the guide.

## 3.2 Downloading the Models

We strongly recommend downloading the models in advance to avoid potential issues during the benchmark runs. To simplify the downloading process, we provide a helper script `download_models.py` that downloads all three models using your obtained HuggingFace access token. Please follow the steps below to download the models.

```sh
# NOTE: make sure no venv is activated in the current terminal
# navigate to the this directory
cd flashattention-t-artifact/3-table2-end2end-generative-accuracy

# export your HuggingFace access token
# example: export HF_TOKEN=hf_1234123412341234abcdabcdabcdabcdab
export HF_TOKEN=$hf_YOURTOKEN

# download the models using the helper script
# NOTE: This script will exit with an error if the HF_TOKEN envvar is not set or not valid
python3 ./download_models.py
```

The script will download the three models into the following subdirectories under the current directory:

- `./Llama-2-13b-hf/` for the Llama2 13B model
- `./Mistral-Nemo-Instruct-2407/` for the Mistral Nemo model
- `./Qwen3-14B-Base/` for the Qwen3 14B model

# 4. Running the MMLU Benchmark

After setting up the environment and downloading the models, you can run the MMLU benchmark by following the steps below:

## 4.1 Benchmark with Original FlashAttention-2

```sh
cd flashattention-t-artifact/3-table2-end2end-generative-accuracy
# 1. activate the .venv-orig-torch venv
source ../2-figure11-synthetic-precision/.venv-orig-torch/bin/activate
lm_eval --model hf --model_args pretrained=./Llama-2-13b-hf --tasks mmlu > mmlu_llama2_13b_fa2_results.log
lm_eval --model hf --model_args pretrained=./Mistral-Nemo-Instruct-2407 --tasks mmlu > mmlu_mistral_nemo_fa2_results.log
lm_eval --model hf --model_args pretrained=./Qwen3-14B-Base --tasks mmlu > mmlu_qwen3_14b_fa2_results.log
# 2. deactivate the venv after the benchmark is complete
deactivate
```

Each of the above commands may take around 10 minutes to complete. Once finished, the commands will generate the following log files under the current directory: 

- `mmlu_llama2_13b_fa2_results.log`
- `mmlu_mistral_nemo_fa2_results.log`
- `mmlu_qwen3_14b_fa2_results.log`

At the end of each log file, you should see the MMLU benchmark results similar to the below table:

```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6394|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5785|±  |0.0068|
| - other          |      2|none  |      |acc   |↑  |0.7107|±  |0.0079|
| - social sciences|      2|none  |      |acc   |↑  |0.7439|±  |0.0076|
| - stem           |      2|none  |      |acc   |↑  |0.5582|±  |0.0085|
```

The value in the `Value` column under the `mmlu` row is the overall MMLU accuracy that corresponds to the results shown in Table 2 of the FlashAttention-T paper.


## 4.2 Benchmark with FlashAttention-T

```sh
cd flashattention-t-artifact/3-table2-end2end-generative-accuracy
# 1. activate the .venv-fat-torch venv
source ../2-figure11-synthetic-precision/.venv-fat-torch/bin/activate
lm_eval --model hf --model_args pretrained=./Llama-2-13b-hf --tasks mmlu > mmlu_llama2_13b_fat_results.log
lm_eval --model hf --model_args pretrained=./Mistral-Nemo-Instruct-2407 --tasks mmlu > mmlu_mistral_nemo_fat_results.log
lm_eval --model hf --model_args pretrained=./Qwen3-14B-Base --tasks mmlu > mmlu_qwen3_14b_fat_results.log
# 2. deactivate the venv after the benchmark is complete
deactivate
```

Each of the above commands may take around 10 minutes to complete. Once finished, the commands will generate the following log files under the current directory:

- `mmlu_llama2_13b_fat_results.log`
- `mmlu_mistral_nemo_fat_results.log`
- `mmlu_qwen3_14b_fat_results.log`

At the end of each log file, you should see the MMLU benchmark results similar to the below table:

```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6394|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5785|±  |0.0068|
| - other          |      2|none  |      |acc   |↑  |0.7107|±  |0.0079|
| - social sciences|      2|none  |      |acc   |↑  |0.7439|±  |0.0076|
| - stem           |      2|none  |      |acc   |↑  |0.5582|±  |0.0085|
```

The value in the `Value` column under the `mmlu` row is the overall MMLU accuracy that corresponds to the results shown in Table 2 of the FlashAttention-T paper.


