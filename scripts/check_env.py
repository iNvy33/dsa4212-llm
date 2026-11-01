"""Print the GPU/software environment and check whether torch.compile works.

Usage: .venv\\Scripts\\python.exe scripts\\check_env.py
"""
import platform
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def triton_version() -> str:
    try:
        import triton
        return triton.__version__
    except ImportError:
        return "not installed"


class TinyBlock(nn.Module):
    """Small attention + FFN block, enough to exercise SDPA, matmuls and GELU."""

    def __init__(self, d: int = 128, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.RMSNorm(d)
        self.fc1 = nn.Linear(d, 2 * d, bias=False)
        self.fc2 = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def check_compile() -> tuple[bool, str]:
    torch.manual_seed(0)
    model = TinyBlock().cuda()
    x = torch.randn(4, 64, 128, device="cuda")

    def step(m):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x)
        out.float().pow(2).mean().backward()
        return out.detach().float()

    try:
        eager = step(model)
        compiled = torch.compile(model)
        t0 = time.perf_counter()
        out = step(compiled)
        torch.cuda.synchronize()
        warmup = time.perf_counter() - t0
    except Exception as e:  # report any compile failure rather than crash
        return False, f"{type(e).__name__}: {e}".splitlines()[0]
    max_diff = (out - eager).abs().max().item()
    ok = max_diff < 0.1  # bf16 tolerance
    return ok, f"warm-up {warmup:.1f}s, max |compiled - eager| = {max_diff:.2e}"


def main():
    print(f"Python        : {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA (torch)  : {torch.version.cuda}")
    print(f"cuDNN         : {torch.backends.cudnn.version()}")
    print(f"Driver        : {driver_version()}")
    print(f"Triton        : {triton_version()}")
    if not torch.cuda.is_available():
        print("CUDA available: False")
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU           : {props.name} ({props.total_memory / 2**30:.1f} GiB)")
    print(f"bf16 support  : {torch.cuda.is_bf16_supported()}")
    print(f"torch.optim.Muon: {hasattr(torch.optim, 'Muon')}")
    ok, detail = check_compile()
    print(f"torch.compile : {'works' if ok else 'FAILED'} ({detail})")


if __name__ == "__main__":
    main()
