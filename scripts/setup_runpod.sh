#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available. Use a GPU Pod with the RunPod PyTorch template.')
print('Torch:',torch.__version__,'CUDA:',torch.version.cuda)
print('GPU:',torch.cuda.get_device_name(0))
print('VRAM GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)
try: print('Native BF16:',torch.cuda.is_bf16_supported(including_emulation=False))
except TypeError: raise SystemExit('Use PyTorch >=2.6. Do not replace CUDA Torch with a CPU wheel.')
PY
python -m pip install 'datasets==3.6.0' 'tokenizers==0.21.1' 'matplotlib>=3.8,<4' 'nbformat>=5.10,<6'
python - <<'PY'
import datasets,tokenizers,matplotlib,torch
x=torch.randn(32,32,device='cuda',requires_grad=True)
y=(x@x).square().mean();y.backward();torch.cuda.synchronize()
print('CUDA matmul and backward passed; dependencies ready.')
PY
