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

assert ORIGINAL_TORCH_VENV_PATH.exists(), f"Original Torch venv path does not exist: {ORIGINAL_TORCH_VENV_PATH}, make sure you have finished setting up the synthetic precision evaluation environment first."
assert FAT_TORCH_VENV_PATH.exists(), f"FAT Torch venv path does not exist: {FAT_TORCH_VENV_PATH}, make sure you have finished setting up the synthetic precision evaluation environment first."


def install_deps_for_venv(venv_pip_path):
    INSTALL_LIST = [
        "lm-eval==0.4.9.1",
        "huggingface_hub",
        "hf_transfer"
    ]
    cmd_str = f"{venv_pip_path} install " + " ".join(INSTALL_LIST)
    subprocess.run(cmd_str, shell=True, executable='/bin/bash')

if __name__ == "__main__":
    install_deps_for_venv(ORIGINAL_TORCH_VENV_PIP_PATH)
    install_deps_for_venv(FAT_TORCH_VENV_PIP_PATH)