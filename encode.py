"""MiniMax-H3 video VAE: encode a reference clip to condition latents.

The documented modular path — `MiniMaxH3Ref2VASetupStep` followed by
`MiniMaxH3Ref2VAReferenceEncoderStep`, assembled into a `MiniMaxH3ModularPipeline`
— exactly as the Ref2VA pipeline does it before handing latents to the transformer.
Only the `vae` and `audio_vae` components are loaded; the 33B transformer and the
text encoder are never downloaded.

    python encode.py                      # jellyfish clip, 124 frames, 3 timed runs
    python encode.py --runs 5 --frames 124

Prints one line per run and the median, and writes out/latents.pt.
"""
import argparse
import json
import os
import statistics
import time

import av
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "MiniMaxAI/MiniMax-H3"
REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def decode_video(path, n_frames):
    """First n_frames RGB frames of an mp4, and its fps. PyAV, no ffmpeg binary needed."""
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        frames = []
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    return np.stack(frames), fps


def build_pipeline(dtype):
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VAReferenceEncoderStep

    blocks = SequentialPipelineBlocks.from_blocks_dict({
        "setup": MiniMaxH3Ref2VASetupStep(),
        "reference_encoder": MiniMaxH3Ref2VAReferenceEncoderStep(),
    })
    pipe = blocks.init_pipeline(REPO)
    pipe.load_components(names=["vae", "audio_vae"], revision=REVISION, dtype=dtype)
    pipe.vae.to("cuda").eval()
    pipe.audio_vae.to("cuda").eval()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "data", "jellyfish.mp4"))
    ap.add_argument("--frames", type=int, default=124, help="latent-aligned count (17n+5)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", default=os.path.join(HERE, "out", "latents.pt"))
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_FPS

    t0 = time.perf_counter()
    raw, fps = decode_video(args.video, n_frames=int(args.frames * 1.3) + 8)
    # The library's own normalizer (24 fps, canvas 768 short edge / 768x1344 max),
    # run once up front so every timed run sees identical, already-normalized input.
    frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
        raw, fps, args.frames, 32, 768, 768 * 1344, float(MINIMAX_H3_FPS))
    print(f"input: {raw.shape} @ {fps:.2f} fps -> {frames.shape} @ {MINIMAX_H3_FPS} fps "
          f"({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    pipe = build_pipeline(dtype)
    print(f"model: vae + audio_vae loaded in {time.perf_counter() - t0:.1f}s "
          f"({args.dtype}, {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.0f}M vae params)")

    reference = MiniMaxH3VideoReference(frames=frames, fps=float(MINIMAX_H3_FPS))
    walls, latents = [], None
    with torch.no_grad():
        for i in range(args.warmup + args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(references=[reference], num_frames=args.frames,
                       output="condition_latents")
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tag = "warmup" if i < args.warmup else f"run {i - args.warmup + 1}"
            print(f"  {tag:8s} {dt:.3f}s")
            if i >= args.warmup:
                walls.append(dt)
            latents = out[0] if isinstance(out, (list, tuple)) else out

    lat = latents[0] if isinstance(latents, list) else latents
    med = statistics.median(walls)
    print(f"encode: median {med:.3f}s per {args.frames}-frame clip over {len(walls)} runs "
          f"({args.frames / med:.1f} frames/s), peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"latents: {tuple(lat.shape)} {lat.dtype}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(lat.cpu(), args.out)
    with open(os.path.join(os.path.dirname(args.out), "timing.json"), "w") as fh:
        json.dump({"median_s": med, "runs_s": walls, "frames": args.frames,
                   "dtype": args.dtype, "latent_shape": list(lat.shape)}, fh, indent=1)


if __name__ == "__main__":
    main()
