import os
import sys
import subprocess
import pathlib
import re

_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve().expanduser().absolute()

ORIGINAL_TORCH_VENV_PATH = _THIS_SCRIPT_DIR / ".venv-orig-torch"
ORIGINAL_TORCH_VENV_PIP_PATH = ORIGINAL_TORCH_VENV_PATH / "bin" / "pip3"

FAT_TORCH_SOURCE_DIR_PATH = _THIS_SCRIPT_DIR / "pytorch"

FAT_TORCH_VENV_PATH = _THIS_SCRIPT_DIR / ".venv-fat-torch"
FAT_TORCH_VENV_PIP_PATH = FAT_TORCH_VENV_PATH / "bin" / "pip3"
FAT_TORCH_VENV_PYTHON_PATH = FAT_TORCH_VENV_PATH / "bin" / "python3"
FAT_TORCH_VENV_ACTIVATE_PATH = FAT_TORCH_VENV_PATH / "bin" / "activate"

FATMAX16_TORCH_VENV_PATH = _THIS_SCRIPT_DIR / ".venv-fatmax16-torch"
FATMAX16_TORCH_VENV_PIP_PATH = FATMAX16_TORCH_VENV_PATH / "bin" / "pip3"
FATMAX16_TORCH_VENV_PYTHON_PATH = FATMAX16_TORCH_VENV_PATH / "bin" / "python3"
FATMAX16_TORCH_VENV_ACTIVATE_PATH = FATMAX16_TORCH_VENV_PATH / "bin" / "activate"


def configure_fa_t_macros(
    updates
):
    header_path = FAT_TORCH_SOURCE_DIR_PATH / "third_party" / "flash-attention" / "csrc" / "flash_attn" / "src" / "softmax_mma.h"
    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")
    content = header_path.read_text(encoding="utf-8")
    for macro, new_value in updates:
        content = re.sub(rf'#define\s+{macro}\s+\d+', f'#define {macro} {new_value}', content)

    header_path.write_text(content, encoding="utf-8")


def reset_fat_pytorch_source():
    build_dir = FAT_TORCH_SOURCE_DIR_PATH / "build"
    if build_dir.exists() and build_dir.is_dir():
        subprocess.run(f"rm -rf {build_dir}", shell=True, executable='/bin/bash')
    
    updates = [
        ("USE_DEFAULT_NEGINF_MASK", "0"),
        ("USE_ACC_S_LEVEL_INF_MASKING", "0"),
        ("USE_MMA_SOFTMAX", "1"),
        ("USE_BINARY_TREE_MAX", "1"),
        ("USE_DEFAULT_MAX", "0"),
        ("LOOP1_USE_ACCS_SCALE_MMA_ONLY", "0"),
        ("LOOP1_USE_ACCS_SCALE_SIMT_ONLY", "0"),
        ("LOOP1_USE_ACCS_SCALE_ILP_VERT", "0"),
        ("LOOP1_USE_ACCS_SCALE_ILP_HORI", "1"),
        ("LOOP1_ACCS_ILP_HORI_RATIO", "4"),
        ("LOOP2_USE_ACCS_SCALE_MMA_ONLY", "0"),
        ("LOOP2_USE_ACCS_SCALE_SIMT_ONLY", "0"),
        ("LOOP2_USE_ACCS_SCALE_ILP_VERT", "0"),
        ("LOOP2_USE_ACCS_SCALE_ILP_HORI", "1"),
        ("LOOP2_ACCS_ILP_HORI_RATIO", "4"),
    ]
    configure_fa_t_macros(updates)

def set_fat_max16_pytorch_source():
    updates = [
        ("USE_DEFAULT_NEGINF_MASK", "0"),
        ("USE_ACC_S_LEVEL_INF_MASKING", "0"),
        ("USE_MMA_SOFTMAX", "1"),
        ("USE_BINARY_TREE_MAX", "0"),
        ("USE_DEFAULT_MAX", "1"),
        ("LOOP1_USE_ACCS_SCALE_MMA_ONLY", "0"),
        ("LOOP1_USE_ACCS_SCALE_SIMT_ONLY", "1"),
        ("LOOP1_USE_ACCS_SCALE_ILP_VERT", "0"),
        ("LOOP1_USE_ACCS_SCALE_ILP_HORI", "0"),
        ("LOOP1_ACCS_ILP_HORI_RATIO", "0"),
        ("LOOP2_USE_ACCS_SCALE_MMA_ONLY", "0"),
        ("LOOP2_USE_ACCS_SCALE_SIMT_ONLY", "1"),
        ("LOOP2_USE_ACCS_SCALE_ILP_VERT", "0"),
        ("LOOP2_USE_ACCS_SCALE_ILP_HORI", "0"),
        ("LOOP2_ACCS_ILP_HORI_RATIO", "0"),
    ]
    configure_fa_t_macros(updates)

def make_venv(venv_path: pathlib.Path) -> None:
    os.system(f"python3.12 -m venv {venv_path}")



if __name__ == "__main__":

    
    ### orig torch venv setup
    # 1. create venv with original torch v2.7.0
    make_venv(ORIGINAL_TORCH_VENV_PATH)
    
    # 2. install original torch v2.7.0
    os.system(f"{ORIGINAL_TORCH_VENV_PIP_PATH} install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128")
    os.system(f"{ORIGINAL_TORCH_VENV_PIP_PATH} install parse")
    os.system(f"{ORIGINAL_TORCH_VENV_PIP_PATH} install numpy")
    os.system(f"{ORIGINAL_TORCH_VENV_PIP_PATH} install matplotlib")
    ### ILP fat venv setup
    reset_fat_pytorch_source()

    # 1. create venv with torch v2.7.0 + FA-T
    make_venv(FAT_TORCH_VENV_PATH)

    # 2. install build dependencies
    os.system(f"{FAT_TORCH_VENV_PIP_PATH} install -r {FAT_TORCH_SOURCE_DIR_PATH / 'requirements.txt'}")
    os.system(f"{FAT_TORCH_VENV_PIP_PATH} install mkl-static mkl-include")
    
    # 3. build and install FA-T torch
    cmd = (
        f"source {FAT_TORCH_VENV_ACTIVATE_PATH} && "
        f"cd {FAT_TORCH_SOURCE_DIR_PATH} && "
        # optional envvars: USE_FBGEMM=0 USE_ROCM=0 USE_SYSTEM_NCCL=1
        f"USE_FBGEMM=0 USE_ROCM=0 USE_SYSTEM_NCCL=1 _GLIBCXX_USE_CXX11_ABI=1 {FAT_TORCH_VENV_PYTHON_PATH} setup.py install"
    )
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')

    ### Max16-only fat venv setup
    set_fat_max16_pytorch_source()

    # 1. create venv with torch v2.7.0 + FA-T
    make_venv(FATMAX16_TORCH_VENV_PATH)
    
    # 2. install build dependencies
    os.system(f"{FATMAX16_TORCH_VENV_PIP_PATH} install -r {FAT_TORCH_SOURCE_DIR_PATH / 'requirements.txt'}")
    os.system(f"{FATMAX16_TORCH_VENV_PIP_PATH} install mkl-static mkl-include")

    # 3. build and install FA-T torch
    cmd = (
        f"source {FATMAX16_TORCH_VENV_ACTIVATE_PATH} && "
        f"cd {FAT_TORCH_SOURCE_DIR_PATH} && "
        # optional envvars: USE_FBGEMM=0 USE_ROCM=0 USE_SYSTEM_NCCL=1
        f"USE_FBGEMM=0 USE_ROCM=0 USE_SYSTEM_NCCL=1 _GLIBCXX_USE_CXX11_ABI=1 {FATMAX16_TORCH_VENV_PYTHON_PATH} setup.py install"
    )
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')

    reset_fat_pytorch_source()

    