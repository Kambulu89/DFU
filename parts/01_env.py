# %% [markdown]
# ## 2–4. Ambiente Kaggle, dependências e hardware
#
# Esta célula funciona com Internet ligada (instala o que faltar) ou desligada (procura *wheels* em `/kaggle/input`).
# Nunca usa credenciais. Em Kaggle os resultados vão para `/kaggle/working`; fora do Kaggle usam-se as variáveis
# `DFU_INPUT` (dados) e `DFU_WORK` (saídas) ou pastas locais.

# %%
import os, sys, platform, subprocess, importlib, json, time, math, random, shutil, hashlib, glob, warnings, itertools, copy, socket
from pathlib import Path
from collections import OrderedDict, defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

IS_KAGGLE = Path("/kaggle/input").exists()
INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path(os.environ.get("DFU_INPUT", "."))
WORK = Path("/kaggle/working") if IS_KAGGLE else Path(os.environ.get("DFU_WORK", "./working"))
WORK.mkdir(parents=True, exist_ok=True)


def has_internet(host: str = "pypi.org", port: int = 443, timeout: float = 4.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


INTERNET = has_internet()


def _version_ok(have: str, need: str | None) -> bool:
    if need is None:
        return True
    from packaging.version import Version
    return Version(have.split("+")[0]) >= Version(need)


def find_wheels(stem: str, max_depth: int = 4) -> list[Path]:
    """Procura *wheels* locais (Kaggle Datasets) sem varrer ficheiros de dados."""
    out: list[Path] = []
    if not INPUT_ROOT.exists():
        return out
    base_depth = len(INPUT_ROOT.parts)
    for dirpath, dirnames, filenames in os.walk(INPUT_ROOT):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
        out += [Path(dirpath) / f for f in filenames if f.endswith(".whl") and f.lower().startswith(stem.lower())]
    return out


def ensure_package(import_name: str, pip_name: str | None = None, min_version: str | None = None) -> None:
    pip_name = pip_name or import_name
    try:
        mod = importlib.import_module(import_name)
        if _version_ok(getattr(mod, "__version__", "0"), min_version):
            return
    except ImportError:
        pass
    spec = f"{pip_name}>={min_version}" if min_version else pip_name
    if INTERNET:
        cmd = [sys.executable, "-m", "pip", "install", "-q", spec]
    else:
        wheels = find_wheels(pip_name.replace("-", "_"))
        if not wheels:
            raise RuntimeError(
                f"Falta o pacote '{pip_name}' e não há Internet nem wheel em /kaggle/input. "
                f"Ative a Internet no notebook ou adicione um Dataset com o wheel de {pip_name}."
            )
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", str(wheels[0])]
    print("A instalar:", " ".join(cmd[3:]))
    subprocess.run(cmd, check=True)
    importlib.invalidate_caches()
    importlib.import_module(import_name)


ensure_package("packaging")
ensure_package("segmentation_models_pytorch", "segmentation-models-pytorch", "0.5.0")  # UPerNet/Segformer exigem >= 0.5
ensure_package("timm", "timm", "0.9.0")
ensure_package("cv2", "opencv-python-headless")
for _imp, _pip in [("pandas", "pandas"), ("matplotlib", "matplotlib"), ("scipy", "scipy"), ("tqdm", "tqdm")]:
    ensure_package(_imp, _pip)

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
import matplotlib

if not IS_KAGGLE and os.environ.get("DFU_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

try:
    from IPython.display import display
except ImportError:  # execução fora de Jupyter
    display = print

cv2.setNumThreads(0)  # evita sobre-subscrição com os workers do DataLoader

# %% [markdown]
# ### Hardware e otimização de recursos
# Deteta CUDA, nome/VRAM da GPU e decide *mixed precision*. **T4/P100 não suportam bfloat16 eficiente**, por isso usa-se fp16 com
# `GradScaler`. O *batch size* efetivo é fixo (`CFG.batch_size`); o *micro-batch* real é descoberto por sondagem de OOM e compensado
# por acumulação de gradientes.

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HW = {
    "device": str(DEVICE),
    "cuda_count": torch.cuda.device_count() if DEVICE.type == "cuda" else 0,
    "gpu_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if DEVICE.type == "cuda" else None,
    "cpu_count": os.cpu_count(),
    "python": platform.python_version(),
    "torch": torch.__version__, "timm": timm.__version__, "smp": smp.__version__,
    "opencv": cv2.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    "internet": INTERNET, "is_kaggle": IS_KAGGLE,
}
USE_AMP = DEVICE.type == "cuda"
print(json.dumps(HW, indent=2))
if DEVICE.type == "cpu":
    print("AVISO: sem GPU. Em Kaggle ative Settings > Accelerator > GPU T4/P100. O modo 'full' é impraticável em CPU.")
