import os
import sys
import subprocess
import pathlib
import re

_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve().expanduser().absolute()

PYTORCH_DIR = _THIS_SCRIPT_DIR / "pytorch"
PYTORCH_SELF_HOSTED_FLASH_API_PATH = PYTORCH_DIR / "aten" / "src" / "ATen" / "native" / "transformers" / "cuda" / "flash_attn" / "flash_api.cpp"

if __name__ == "__main__":
    # clone pytorch v270
    os.system(f"git clone --branch v2.7.0 --recursive https://github.com/pytorch/pytorch.git {PYTORCH_DIR}")

    # clean up the cloned pytorch repo to save space
    os.system(f"rm -rf {PYTORCH_DIR / '.git'}")
    os.system(f"rm -rf {PYTORCH_DIR / '.github'}")
    os.system(f"rm -rf {PYTORCH_DIR / '.vscode'}")

    # remove the original flash attention repo
    os.system(f"rm -rf {PYTORCH_DIR / 'third_party' / 'flash-attention'}")
    
    # replace it with flashattention-t
    os.system(f"cp -r {_THIS_SCRIPT_DIR / 'flash-attention-t'} {PYTORCH_DIR / 'third_party' / 'flash-attention'}")

    # disable splitkv which is irrelevant to FA-T and may complicate testing
    assert PYTORCH_SELF_HOSTED_FLASH_API_PATH.exists(), f"File not found: {PYTORCH_SELF_HOSTED_FLASH_API_PATH}"
    content = PYTORCH_SELF_HOSTED_FLASH_API_PATH.read_text(encoding="utf-8")
    content_lines = content.splitlines()

    for idx, line in enumerate(content_lines):
        if "return std::make_tuple(softmax_lse_accum, out_accum);" in line:
            content_lines.insert(idx, "    params.num_splits = 1;")
            break

    content = "\n".join(content_lines)
    PYTORCH_SELF_HOSTED_FLASH_API_PATH.write_text(content, encoding="utf-8")