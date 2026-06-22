
# 1. Introduction

Welcome to the artifact package for our PPoPP '26 paper, "FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism". Given the limited timeframe for artifact preparation and the fact that most reviewer concerns centered on the numerical stability and precision of FlashAttention-T, this artifact package contains code and instructions for fully reproducing the three main results reported in the FlashAttention-T paper:

1. **Main results**: performance comparisons between FlashAttention-T and baselines FlashAttention-2/3, Triton, and FlashInfer on A100 and H100 GPUs (Figure 8 in the paper).

2. **Synthetic attention precision evaluation**: numerical precision comparisons between FlashAttention-T, an ablation variant of FlashAttention-T with only the 16-row maximum surrogate enabled, and FlashAttention-2 on A100 GPU using random attention inputs sampled from specific distributions. (Figure 11 in the paper).

3. **End-to-end generative benchmark accuracy evaluation**: MMLU benchmark score comparisons between FlashAttention-T and FlashAttention-2 on A100 GPU (Table 2 in the paper).

We believe these three results cover the most important aspects of our proposed FlashAttention-T attention kernel, including its performance benefits, numerical stability/precision preservation, and end-to-end accuracy impact on generative workloads. They directly support the main claims made during the review process and in the paper:

1. **Implementation details**: FlashAttention-T is a prototype attention kernel library built on FlashAttention-2 and FlashAttention-3. The core modifications implementing the proposed techniques comprise approximately 4K lines of CUDA/C++ code.

2. **Main performance claim**: Experimental results show that FlashAttention-T outperforms the state-of-the-art FlashAttention-2 and FlashAttention-3 with average attention throughput speedups of 1.05–1.17× across diverse attention configurations on Ampere and Hopper GPUs.

3. **Numerical precision claim**: In the synthetic attention precision evaluation, FlashAttention-T achieves an RMSE with respect to the FP64 reference attention output less than 1e-3 and is in the same order of magnitude as FlashAttention-2, with no observed numerical failure.

4. **End-to-end generative benchmark accuracy claim**: In the MMLU benchmark with the three evaluated LLMs (Llama2 13B, Mistral NeMo, Qwen3 14B), FlashAttention-T achieves scores nearly identical to the baseline FlashAttention-2, with no numerical failures detected.

# 2. Getting Started

As we've prepared very detailed step-by-step instructions with helper scripts that reduce manual efforts for reproducing each of the three main results in their respective subdirectories, and we **strongly recommend** starting all the evaluations with a **clean artifact state** (i.e., starting from the original downloaded artifact package without any modifications), we mainly introduce the following in this section:

- Evaluation system requirements
- Deploying evaluation systems with runpod.io
- An overview of the directory structure of this artifact package

## 2.1 Evaluation System Requirements

Conducting the evaluations in this artifact package requires two machines satisfying the following system requirements:

Machine 1 (The "A100 Machine"):

- System: Modern Linux distros, bare metal/container environment
- GPU: NVIDIA A100 80GB SXM4
- CUDA Version: 12.8/12.9
- Python Version: 3.11/3.12
- CPU: x86_64 CPU with sufficient cores to speed up compilation
- RAM: Recommend at least 64GB of RAM to avoid unexpected OOM error during compilation and model loading
- Disk: At least 400GB free disk space


Machine 2 (The "H100 Machine"):

- System: Modern Linux distros, bare metal/container environment
- GPU: NVIDIA H100 80GB PCIe/SXM5
- CUDA Version: 12.8/12.9
- Python Version: 3.11/3.12
- CPU: x86_64 CPU with sufficient cores to speed up compilation
- RAM: Recommend at least 64GB of RAM to avoid unexpected OOM error during compilation
- Disk: At least 200GB free disk space

In case you do not have access to physical machines satisfying the above requirements, we **strongly recommend** using [runpod.io](https://runpod.io/) to rent cloud GPU instances that meet the requirements for convenience (since we have tested this artifact package there). Please refer to the [next subsection](#2.2-deploying-evaluation-systems-with-runpodio) for detailed instructions.

In addition, an important note on using runpod GPUs for conducting the evaluations is that the absolute throughput figures on runpod GPUs may be lower than the reported figures in our paper, though the relative speedups should be similar and this does not affect the conclusions of the paper. This is because the power limits of the runpod GPUs can be stricter than the GPUs we tested on. (e.g. H100 PCIe: 310W on runpod vs 350W ours)

## 2.2 Deploying Evaluation Systems with runpod.io

### 2.2.1 Creating an Account on runpod.io

1. Navigate to the runpod.io website: https://runpod.io/ in your web browser.

2. Click on the "Sign in" button at the top right corner of the homepage. You can sign up using your Github or Google account if you'd like, or you can sign up with your email address by clicking on the "Sign up with Email" link at the bottom of the sign-in form. Follow the instructions to complete the registration process.

3. After signing up, you need to first **configure your SSH keys** so you could ssh into the rented instances later (NOTE: runpod.io instances does NOT accept password-based ssh login so this step is mandatory). To do so, please first click the "Setting" tab in the left sidebar after logging into your account. The site should now jump to the https://console.runpod.io/user/settings page.

4. Scroll down to the "SSH Public Keys" section, and copy-paste your SSH public key into the text box (ED25519 keys are recommended, e.g., `~/.ssh/id_ed25519.pub`). If you do not have an SSH key pair yet, please follow the instructions in [this guide](https://www.ssh.com/academy/ssh/keygen) to generate one. After pasting your public key, click on the "Update public key" button to save it.

### 2.2.2 Deploying the A100 Machine

1. Navigate to the runpod.io website: https://runpod.io/ in your web browser, and log into your account.

2. Click on the "Pods" tab in the left sidebar. The site should now jump to the https://console.runpod.io/deploy page, and you will see a list of available instances.

3. Look for the instance named "A100 SXM" under the "NVIDIA Previous Gen" section. Click on it to enter the deployment configuration page.

4. In the deployment configuration page, set the following options:

- Pod Name: Set a name you prefer, or leave it as the default random value.
- Pod Template: First select "Change Template", choose "Runpod Pytorch 2.8.0". Then, after returning to the deployment configuration page, click on the "Edit" button next to the "Change Template" button, and set:
   - Container Disk: 300 GB
   - Volume Disk: 100 GB
   - Volume Mount Path: leave unchanged (`/workspace`)
   - Expose HTTP Ports: leave unchanged (`8888`)
   - Expose TCP Ports: leave unchanged (`22` for ssh)
   - Click on "Set Overrides" to save the changes and return to the deployment configuration page.
- GPU Count: 1
- Instance Pricing: Select "On-Demand"
- Encrypt Volume: Unchecked
- SSH Terminal Access: Checked
- Start Jupyter Notebook: Leave as default

Here the disk size configurations are crucial to ensure sufficient disk space for conducting the evaluations in this artifact package, so please make sure to set them as above. After the instance is deployed, the "Container Disk" will be used as the root filesystem mounted at `/`, while the "Volume Disk" will be mounted at `/workspace`.

5. After configuring the above options, click on the "Deploy On-Demand" button at the bottom of the page. The page will now automatically jump to the management page of your newly deployed instance. It may take around 30 seconds to a few minutes for the instance to start, so please be patient.

6. Once the instance is running, you should now see the management options fully loaded. Please adapt the command shown in the "SSH over exposed TCP" section to ssh connect to the instance. For example, the command may look like:

```sh
ssh root@213.173.102.6 -p 44320 -i ~/.ssh/id_ed25519
```

A detailed taxonomy of this ssh commmand is as follows:

- `root`: The default username for runpod.io instances.
- `-p 44320`: The port number for ssh connection, which may vary for different instances. Please make sure to use the port number shown in your instance management page.
- `-i ~/.ssh/id_ed25519`: The path to your SSH private key that corresponds to the public key you added in the previous subsection.

After successful ssh login, you should now see a terminal prompt similar to:

```sh
 _____                             _ 
|  __ \                           | |
| |__) |   _ _ __  _ __   ___   __| |
|  _  / | | | '_ \| '_ \ / _ \ / _` |
| | \ \ |_| | | | | |_) | (_) | (_| |
|_|  \_\__,_|_| |_| .__/ \___/ \__,_|
                  | |                
                  |_|                
For detailed documentation and guides, please visit:
https://docs.runpod.io/ and https://blog.runpod.io/


root@a2ea5d1b3b00:/# 
```

And a run of `nvidia-smi` should show the A100 SXM GPU information.

### 2.2.3 Deploying the H100 Machine

All the steps for deploying the H100 machine are the same to those for the A100 machine, except for the instance selection in Step 3, and storage configuration in Step 4. Specifically, please set the following options in the deployment configuration page:

#### Instance Selection

Please look for the instance named "H100 PCIe" under the "NVIDIA Latest Gen" section. Click on it to enter the deployment configuration page. In addition, "H100 SXM" is also viable for conducting the evaluations, but the concrete performance figures may be different from those reported in the paper since the H100 SXM has higher SM count, memory bandwidth, and power limit than the H100 PCIe.

#### Storage Configuration

- Pod Name: Set a name you prefer, or leave it as the default random value.
- Pod Template: First select "Change Template", choose "Runpod Pytorch 2.8.0". Then, after returning to the deployment configuration page, click on the "Edit" button next to the "Change Template" button, and set:
   - Container Disk: 100 GB
   - Volume Disk: 100 GB
   - Volume Mount Path: leave unchanged (`/workspace`)
   - Expose HTTP Ports: leave unchanged (`8888`)
   - Expose TCP Ports: leave unchanged (`22` for ssh)
   - Click on "Set Overrides" to save the changes and return to the deployment configuration page.
- GPU Count: 1
- Instance Pricing: Select "On-Demand"
- Encrypt Volume: Unchecked
- SSH Terminal Access: Checked
- Start Jupyter Notebook: Leave as default

The only difference here compared to the A100 machine is that the "Container Disk" is set to 100 GB since the H100 machine is not used for conducting the end-to-end generative accuracy evaluation that requires large model downloads.


### 2.2.4 Uploading and Deflating the Artifact Package

We provide bzip2 compressed tarball (`.tbz2`) of the artifact package. If you have set up both the A100 and H100 machines on runpod.io, you can upload the artifact package to either of the two machines and then use `scp` to transfer the package to the two machines from your local machine:

```sh
scp -P <SSH_PORT> flashattention-t-artifact.tbz2 root@<MACHINE_IP>:/root/
```
Here `<SSH_PORT>` and `<MACHINE_IP>` should be replaced with the actual ssh port number and IP address of the target machine, shown in the "SSH over exposed TCP" section of the instance management page. Note that we **strongly recommend** uploading and deflating the artifact package under the `/root/` directory on both machines to avoid disk space issues. (The `/workspace/` directory disk space on both machines will mainly be used for caching `pip` packages and downloading models during the evaluations.)

Upon uploading the artifact package to both machines, you can ssh into each machine and deflate the package with the following command:

```sh
cd /root/
tar -xjf flashattention-t-artifact.tbz2
```

This will create a directory `flashattention-t-artifact` under `/root/` containing all the files in this artifact package.

### 2.2.5 Terminating the Instances After Evaluation Completion

After you have completed the evaluations, please make sure to **stop and terminate** the rented instances on runpod.io to avoid unnecessary charges. To do so, please navigate to the "Pods" tab in the left sidebar, click on the instance you want to terminate to open its management page, then:

1. First click on the "Stop" button to stop the instance. This will cause the "Container Disk" to be released, while the "Volume Disk" will be retained for future use.

2. Then click on the "Terminate" button to completely terminate the instance. This will further release the "Volume Disk". This step is crucial to avoid incurring additional charges for the persistent volume storage.

## 2.3 Directory Structure Overview

The directory/file structure of this artifact package is as follows:

```
flashattention-t-artifact                 <- Root dir of the artifact package
├── 1-figure8-main-results                <- Subdir for evaluating the main performance results (Figure 8)
├── 2-figure11-synthetic-precision        <- Subdir for evaluating the synthetic attention precision results (Figure 11)
├── 3-table2-end2end-generative-accuracy  <- Subdir for evaluating the end-to-end generative benchmark accuracy results (Table 2)
└── README.md                             <- A copy of this file
```

Inside each of the three subdirectories, you will find complete code, helper scripts, and detailed step-by-step instructions for reproducing the corresponding results reported in the FlashAttention-T paper.

# 3. Step-by-Step Instructions for Evaluating the Artifact

Please refer to the README.md files under each subdirectory in the artifact package for detailed step-by-step instructions on how to evaluate each of the three main results reported in the FlashAttention-T paper:

- [Main Performance Results Evaluation (Figure 8)](./1-figure8-main-results/README.md) (located under the `flashattention-t-artifact/1-figure8-main-results` directory). This README also contains a brief overview of the FlashAttention-T implementation details on both Ampere and Hopper GPUs in its Section 1.1.
- [Synthetic Attention Precision Evaluation (Figure 11)](./2-figure11-synthetic-precision/README.md) (located under the `flashattention-t-artifact/2-figure11-synthetic-precision` directory).
- [End-to-End Generative Benchmark Accuracy Evaluation (Table 2)](./3-table2-end2end-generative-accuracy/README.md) (located under the `flashattention-t-artifact/3-table2-end2end-generative-accuracy` directory).
