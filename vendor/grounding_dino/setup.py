# grounding_dino/setup.py
# coding=utf-8

import glob
import os
import re
import subprocess
from setuptools import find_packages, setup

package_name = "groundingdino"
version = "0.1.0"
cwd = os.path.dirname(os.path.abspath(__file__))

def write_version_file():
    version_path = os.path.join(cwd, "groundingdino", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")

def _detect_cuda_version(cuda_home: str | None):
    if not cuda_home:
        return None
    try:
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        out = subprocess.check_output([nvcc, "--version"]).decode("utf-8")
        m = re.search(r"release (\d+\.\d+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None

# build switches
BUILD_CUDA = os.getenv("DINO_BUILD_CUDA", "1") == "1"
BUILD_ALLOW_ERRORS = os.getenv("DINO_BUILD_ALLOW_ERRORS", "1") == "1"

CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build GroundingDINO C++/CUDA extension. "
    "Installation can continue, but ops-dependent functionality/performance may be limited.\n"
)

def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    except Exception as e:
        raise RuntimeError(
            "Torch is required to build GroundingDINO extensions. "
            "Install torch/torchvision first (pinned by uv/pyproject.toml), then reinstall."
        ) from e

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    sources = [main_source] + sources

    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    use_cuda = (
        CUDA_HOME is not None
        and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ)
        and len(source_cuda) > 0
    )

    if not use_cuda:
        print("CUDA not available (or no .cu sources). Skipping GroundingDINO extension build.")
        return []

    print("Compiling GroundingDINO extension with CUDA")
    sources += source_cuda

    define_macros = [("WITH_CUDA", None)]
    extra_compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_90,code=sm_90",
        ],
    }

    cuda_version = _detect_cuda_version(CUDA_HOME)
    if cuda_version is not None:
        print(f"Detected CUDA version: {cuda_version}")
        if cuda_version >= 12.8:
            extra_compile_args["nvcc"].append("-gencode=arch=compute_120,code=sm_120")

    ext_modules = [
        CUDAExtension(
            "groundingdino._C",
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

def safe_get_extensions():
    try:
        return get_extensions()
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            return []
        raise

# ---- IMPORTANT: PEP517 editable buildでも確実にBuildExtensionを使う ----
cmdclass = {}
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
    cmdclass = {"build_ext": BuildExtension}
except Exception:
    # torch が無いなら拡張はビルドできないが、ここでsetup自体は通してよい
    cmdclass = {}

with open("LICENSE", "r", encoding="utf-8") as f:
    license_text = f.read()

write_version_file()

setup(
    name=package_name,
    version=version,
    author="International Digital Economy Academy, Shilong Liu",
    url="https://github.com/IDEA-Research/GroundingDINO",
    description="open-set object detector",
    license=license_text,
    install_requires=[],  # 依存固定はuv/pyproject.toml側に寄せる
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=safe_get_extensions(),
    cmdclass=cmdclass,
)