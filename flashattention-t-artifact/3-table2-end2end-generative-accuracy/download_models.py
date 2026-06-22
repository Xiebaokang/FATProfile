import os
import sys
import subprocess
import pathlib
import re

_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve().expanduser().absolute()

def get_synth_precision_eval_dir_path():
    artifact_root = _THIS_SCRIPT_DIR.parent
    for child in artifact_root.iterdir():
        if child.is_dir() and "synthetic-precision" in child.name:
            return child
    raise FileNotFoundError("Could not find synthetic precision evaluation directory.")
        
SYNTH_PREC_EVAL_DIR = get_synth_precision_eval_dir_path()

ORIGINAL_TORCH_VENV_PATH = SYNTH_PREC_EVAL_DIR / ".venv-orig-torch"
ORIGINAL_TORCH_VENV_PIP_PATH = ORIGINAL_TORCH_VENV_PATH / "bin" / "pip3"
ORIGINAL_TORCH_VENV_PYTHON_PATH = ORIGINAL_TORCH_VENV_PATH / "bin" / "python3"
ORIGINAL_TORCH_VENV_ACTIVATE_PATH = ORIGINAL_TORCH_VENV_PATH / "bin" / "activate"

FAT_TORCH_VENV_PATH = SYNTH_PREC_EVAL_DIR / ".venv-fat-torch"
FAT_TORCH_VENV_PIP_PATH = FAT_TORCH_VENV_PATH / "bin" / "pip3"
FAT_TORCH_VENV_PYTHON_PATH = FAT_TORCH_VENV_PATH / "bin" / "python3"
FAT_TORCH_VENV_ACTIVATE_PATH = FAT_TORCH_VENV_PATH / "bin" / "activate"

DOWNLOAD_VENV_PATH = _THIS_SCRIPT_DIR / ".venv-download-models"
DOWNLOAD_VENV_PIP_PATH = DOWNLOAD_VENV_PATH / "bin" / "pip3"
DOWNLOAD_VENV_PYTHON_PATH = DOWNLOAD_VENV_PATH / "bin" / "python3"
DOWNLOAD_VENV_ACTIVATE_PATH = DOWNLOAD_VENV_PATH / "bin" / "activate"

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("Please make sure you have set the HF_TOKEN environment variable with your HuggingFace access token before running this script.")

def make_download_venv():
    if not DOWNLOAD_VENV_PATH.exists():
        subprocess.run(f"python3 -m venv {DOWNLOAD_VENV_PATH}", shell=True, executable='/bin/bash')
    cmd_str = f"{DOWNLOAD_VENV_PIP_PATH} install huggingface_hub hf_transfer"
    subprocess.run(cmd_str, shell=True, executable='/bin/bash')

def download_model(venv_activate_path, model_name, save_dir, hf_token):
    cmd = (
        f"source {venv_activate_path} && "
        f"hf download {model_name} --local-dir {save_dir} --token {hf_token}"
    )
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download model {model_name}. Is your HF_TOKEN valid and do you have access to the model?")

DOWNLOADS = [
    (DOWNLOAD_VENV_ACTIVATE_PATH, "Qwen/Qwen3-14B-Base", _THIS_SCRIPT_DIR/"Qwen3-14B-Base", HF_TOKEN),
    (DOWNLOAD_VENV_ACTIVATE_PATH, "mistralai/Mistral-Nemo-Instruct-2407", _THIS_SCRIPT_DIR/"Mistral-Nemo-Instruct-2407", HF_TOKEN),
    (DOWNLOAD_VENV_ACTIVATE_PATH, "meta-llama/Llama-2-13b-hf", _THIS_SCRIPT_DIR/"Llama-2-13b-hf", HF_TOKEN),
]

if __name__ == "__main__":
    make_download_venv()
    # use serial download as parallel download will mess up the ouput
    for download in DOWNLOADS:
        download_model(*download)