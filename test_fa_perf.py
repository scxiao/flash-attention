#!/usr/bin/env python3
"""
Benchmark script to compare Flash Attention implementations with determinism:
- Triton (with deterministic=True)
- Triton (with deterministic=False)
Usage:
    python benchmark_ck.py                       # Run timing benchmarks only
    python benchmark_ck.py --config 0            # Run only config 0
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import torch
import triton
import flash_attn
import pytest

# Import benchmark utilities
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward
#from aiter.ops.triton.attention.mha import flash_attn_varlen_func
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input
from einops import rearrange, repeat

# from flash-attn/tests/test_flash_attn_triton_amd.py
def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask

def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)
    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )
    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k
    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )
    
# -------------------- Kernel Patterns for Identification --------------------
# These patterns identify which backend/kernel is running in rocprof output
KERNEL_PATTERNS = {
    # Triton AMD kernels (from flash_attn_triton_amd/)
    "triton_fwd_prefill": "attn_fwd",
    "triton_fwd_decode": "_fwd_kernel_splitK",
    "triton_bwd_causal": "bwd_kernel_fused_causal",
    "triton_bwd_noncausal": "bwd_kernel_fused_noncausal",
    "triton_bwd_preprocess": "_bwd_preprocess",
    # CK kernels (from csrc/flash_attn_ck/ via composable_kernel)
    "ck_fwd": "ck_tile::FmhaFwdKernel",
    "ck_bwd": "ck_tile::FmhaBwd",
    # Alternative CK patterns (newer versions)
    "ck_fwd_alt": "fmha_fwd",
    "ck_bwd_alt": "fmha_bwd",
}

def print_version_info():
    """Print version information for Python and key dependencies."""
    print("=" * 80)
    print("VERSION INFORMATION")
    print("=" * 80)
    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Flash-Attn:  {flash_attn.__version__}")
    print(f"Triton:      {triton.__version__}")
    # Check if running on ROCm (AMD) or CUDA (NVIDIA)
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print(f"ROCm/HIP:    {torch.version.hip}")
    else:
        print(f"CUDA:        {torch.version.cuda}")
    # Print backend info
    use_triton = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
    print(f"Backend:     {'Triton AMD' if use_triton else 'CK (Composable Kernel)'}")
    print("=" * 80)
# -------------------- Benchmark Settings --------------------
# Data type
dtype = torch.bfloat16
# Device
device = 'cuda'
# Number of benchmark iterations (used by benchmark_forward/backward)
repeats = 30
# Benchmark configurations
#            b,  sq,   h,  h_k, d, causal
configs = [
            (2, 16384,  32, 32, 64,  False),
            (2, 16384,  32, 32, 64,  True),
            (2, 16384,  32, 4, 64,  False),
            (2, 16384,  32, 4, 64,  True),
]
# Output directories
cwd = os.getcwd()
output_dir_name = "profiler_outputs"
output_csv = "benchmark_results.csv"

def calculate_varlen_attention_tflops(batch_size, cu_seqlens_q, cu_seqlens_k, num_heads_q, head_dim_qk, fwd_time_ms, bwd_time_ms, is_causal):
    """Calculate TFLOPs for attention operations for the varlen mode."""
    ss_count = 0
    for batch_idx in range(batch_size):
      s_q = cu_seqlens_q[batch_idx+1].item()-cu_seqlens_q[batch_idx].item()
      s_k = cu_seqlens_k[batch_idx+1].item()-cu_seqlens_k[batch_idx].item()
      ss_count = ss_count + s_q*s_k
    fwd_flops = (0.5 if is_causal else 1.0) * 4 * ss_count * num_heads_q * head_dim_qk / 1e12
    fwd_tflops = fwd_flops / (fwd_time_ms / 1000.0)
    bwd_tflops = (fwd_flops / (bwd_time_ms / 1000.0)) * 2.5
    return fwd_tflops, bwd_tflops

def benchmark_config(batch_size, seqlen, nheads, nheads_k, headdim, causal, verbose_output=False):
    """Benchmark a single configuration."""
    results = {}
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
    query_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen,
        seqlen,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    if verbose_output:
        print(f"\n{'='*80}")
        print(f"Config: batch={batch_size}, seqlen={seqlen}, nheads={nheads}, nheads_k={nheads_k}, "
              f"headdim={headdim}, causal={causal}")
        print(f"{'='*80}")
    # Benchmark both deterministic modes
    det_modes = [(False, 'nondet'), (True, 'det')]
    for i, (deterministic, key) in enumerate(det_modes):
        if verbose_output:
            print(f"\n[{i+1}/{len(det_modes)}] deterministic={deterministic}")
        time.sleep(0.5)
        q_det = q_unpad.clone().detach().requires_grad_(True)
        k_det = k_unpad.clone().detach().requires_grad_(True)
        v_det = v_unpad.clone().detach().requires_grad_(True)
        cu_seqlens_q_det = cu_seqlens_q.clone().detach()
        cu_seqlens_k_det = cu_seqlens_k.clone().detach()
        _, time_fwd = benchmark_forward(
            flash_attn_varlen_func, q_det, k_det, v_det, cu_seqlens_q_det, cu_seqlens_k_det, seqlen, seqlen,
            causal=causal, deterministic=deterministic,
            repeats=repeats, verbose=False,
        )
        time.sleep(0.5)
        _, time_bwd = benchmark_backward(
            flash_attn_varlen_func, q_det, k_det, v_det, cu_seqlens_q_det, cu_seqlens_k_det, seqlen, seqlen,
            causal=causal, deterministic=deterministic,
            repeats=repeats, verbose=False,
        )
        fwd_time_ms = time_fwd.mean * 1e3
        bwd_time_ms = time_bwd.mean * 1e3
        fwd_tflops, bwd_tflops = calculate_varlen_attention_tflops(
            batch_size, cu_seqlens_q_det.clone().detach(), cu_seqlens_k_det.clone().detach(), nheads, headdim, fwd_time_ms, bwd_time_ms, causal
        )
        results[key] = {
            'fwd_time': time_fwd.mean,
            'bwd_time': time_bwd.mean,
            'fwd_tflops': fwd_tflops,
            'bwd_tflops': bwd_tflops,
        }
        if verbose_output:
            print(f"  Forward:  {fwd_time_ms:.3f} ms  |  {fwd_tflops:.2f} TFLOPs/s")
            print(f"  Backward: {bwd_time_ms:.3f} ms  |  {bwd_tflops:.2f} TFLOPs/s")
    return results

def print_summary_table(all_results):
    """Print the benchmark results table."""
    print("\n" + "="*120)
    print("BENCHMARK RESULTS (PyTorch Timer - End-to-End)")
    print(f"Measured with benchmark_forward/backward, {repeats} iterations averaged")
    print("="*120)
    print(f"\n{'Config':<40} | {'Method':<20} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Fwd TF/s':<10} | {'Bwd TF/s':<10}")
    print("─"*120)
    for item in all_results:
        config = item['config']
        batch_size, seqlen, nheads, nheads_k, headdim, causal = config
        config_str = f"B={batch_size} S={seqlen} H={nheads} H_k={nheads_k} D={headdim} C={causal}"
        results = item['results']
        r = results['nondet']
        print(f"{config_str:<40} | {'det=False':<20} | {r['fwd_time']*1e3:<10.3f} | "
              f"{r['bwd_time']*1e3:<10.3f} | {r['fwd_tflops']:<10.2f} | {r['bwd_tflops']:<10.2f}")
        r = results['det']
        print(f"{config_str:<40} | {'det=True':<20} | {r['fwd_time']*1e3:<10.3f} | "
              f"{r['bwd_time']*1e3:<10.3f} | {r['fwd_tflops']:<10.2f} | {r['bwd_tflops']:<10.2f}")
        print("─"*120)

def save_results_csv(all_results):
    """Save benchmark results to CSV file."""
    csv_filename = output_csv
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'TestID', 'Batch', 'SeqLen', 'NumHeads', 'NumHeads_k', 'HeadDim', 'Causal', 'Deterministic',
            'Fwd (ms)', 'Bwd (ms)', 'Fwd TF/s', 'Bwd TF/s'
        ])
        test_id = 1
        for item in all_results:
            config = item['config']
            batch_size, seqlen, nheads, nheads_k, headdim, causal = config
            results = item['results']
            for det_val, key in [(False, 'nondet'), (True, 'det')]:
                r = results[key]
                writer.writerow([
                    test_id, batch_size, seqlen, nheads, nheads_k, headdim,
                    'TRUE' if causal else 'FALSE', 'TRUE' if det_val else 'FALSE',
                    f"{r['fwd_time']*1e3:.3f}", f"{r['bwd_time']*1e3:.3f}",
                    f"{r['fwd_tflops']:.2f}", f"{r['bwd_tflops']:.2f}"
                ])
                test_id += 1
    print(f"\n✅ Results saved to: {csv_filename}")

def main(args):
    """Main benchmark function (no profiling, just timing)."""
    print_version_info()
    # Determine which configs to run
    if args.config is not None:
        config_indices = [args.config]
    else:
        config_indices = list(range(len(configs)))
    all_results = []
    print("\nRunning benchmarks...", end="", flush=True)
    for idx in config_indices:
        if idx >= len(configs):
            print(f"\nConfig index {idx} out of range (max: {len(configs)-1})")
            continue
        batch_size, seqlen, nheads, nheads_k, headdim, causal = configs[idx]
        results = benchmark_config(batch_size, seqlen, nheads, nheads_k, headdim, causal, verbose_output=False)
        all_results.append({
            'config': (batch_size, seqlen, nheads, nheads_k, headdim, causal),
            'results': results
        })
        print(".", end="", flush=True)
    print(" done!\n")
    # Print summary table and save CSV
    print_summary_table(all_results)
    save_results_csv(all_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Flash Attention with kernel profiling")
    parser.add_argument("--config", type=int, default=None,
                        help="Run only a specific config index (0-based). Works with or without --profile")
    args = parser.parse_args()
    main(args)


def output_file_names(batch_size, causal):
    output_file = f"output/batch_{batch_size}_causal_{causal}"

    fwd_file = f"{output_file}_fwd_outputs.pt"
    bwd_file = f"{output_file}_bwd_outputs.pt"

    return fwd_file, bwd_file


# unit test to verify kernel change correctness
#save output flag
pytest_save_output = False

@pytest.mark.parametrize("batch_size, seqlen, nheads, nheads_k, headdim, causal",
configs)
def test_correctness(batch_size, seqlen, nheads, nheads_k, headdim, causal, verbose_output=False):
    torch.manual_seed(1)
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
    query_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen,
        seqlen,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    if verbose_output:
        print(f"\n{'='*80}")
        print(f"Config: batch={batch_size}, seqlen={seqlen}, nheads={nheads}, nheads_k={nheads_k}, "
              f"headdim={headdim}, causal={causal}")
        print(f"{'='*80}")

    q_det = q_unpad.clone().detach().requires_grad_(True)
    k_det = k_unpad.clone().detach().requires_grad_(True)
    v_det = v_unpad.clone().detach().requires_grad_(True)
    cu_seqlens_q_det = cu_seqlens_q.clone().detach()
    cu_seqlens_k_det = cu_seqlens_k.clone().detach()
    deterministic = True

    fwd_out_name, bwd_out_name = output_file_names(batch_size, causal)
    if pytest_save_output:
        if not os.path.isdir(os.path.dirname(fwd_out_name)):
            os.makedirs(os.path.dirname(fwd_out_name))

    y = flash_attn_varlen_func(q_det, k_det, v_det, cu_seqlens_q_det, cu_seqlens_k_det, seqlen, seqlen, causal=causal, deterministic=deterministic)
    if type(y) is tuple:
        y = y[0]

    if pytest_save_output:
        torch.save(y, fwd_out_name)
    else:
        y_ref = torch.load(fwd_out_name)
        torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=0) 

    grad = torch.randn_like(y)
    y.backward(grad, retain_graph=True)
    dq = q_det.grad.clone()
    dk = k_det.grad.clone()
    dv = v_det.grad.clone()

    if pytest_save_output:
        # save bwd outputs
        torch.save({"dq": dq, 
                    "dk": dk, 
                    "dv": dv}, 
                    bwd_out_name)
    else:
        ref_outpus = torch.load(bwd_out_name)
        dq_ref = ref_outpus["dq"]
        dk_ref = ref_outpus["dk"]
        dv_ref = ref_outpus["dv"]
        torch.testing.assert_close(dq, dq_ref, atol=1e-5, rtol=0.015)
        torch.testing.assert_close(dk, dk_ref, atol=1e-5, rtol=0.015)
        torch.testing.assert_close(dv, dv_ref, atol=1e-5, rtol=0.015)        