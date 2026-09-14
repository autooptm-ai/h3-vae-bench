"""MiniMax-H3 video VAE: decode condition latents back to frames.

The other half of the Ref2VA round trip (encode.py is the first). A set of
clips is encoded ONCE, untimed, through the documented modular path
(`MiniMaxH3Ref2VASetupStep` + `MiniMaxH3Ref2VAReferenceEncoderStep`); the
timed loop then runs `MiniMaxH3VideoDecodeStep` -- the 36-layer ViT decoder
of `AutoencoderKLMiniMaxH3` plus the pixel un-normalisation -- once per clip.
Only `vae` (and `audio_vae`, which the encoder step requires) are loaded; the
transformer and the text encoder are never downloaded.

    python decode.py                      # 4 clips x 124 frames, jellyfish
    python decode.py --clips 2 --frames 56

Prints one line per clip and writes out/decode.json (per-clip wall time,
output shape, mean absolute error against the encoder's input frames).
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


def build_encoder(dtype):
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


def build_decoder(dtype, vae):
    """The decode step as its own pipeline, sharing the encoder's loaded vae."""
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

    blocks = SequentialPipelineBlocks.from_blocks_dict({"decode": MiniMaxH3VideoDecodeStep()})
    pipe = blocks.init_pipeline(REPO)
    pipe.update_components(vae=vae)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(HERE, "data", "jellyfish.mp4"))
    ap.add_argument("--frames", type=int, default=124, help="latent-aligned count (17n+5)")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--stride", type=int, default=30, help="source frames between clip starts")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", default=os.path.join(HERE, "out", "decode.json"))
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_FPS

    need = int(args.frames * 1.3) + 8
    t0 = time.perf_counter()
    raw, fps = decode_video(args.video, n_frames=need + args.stride * (args.clips - 1))
    clips = []
    for i in range(args.clips):
        s = i * args.stride
        clips.append(MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            raw[s:s + need], fps, args.frames, 32, 768, 768 * 1344, float(MINIMAX_H3_FPS)))
    print(f"input: {raw.shape} @ {fps:.2f} fps -> {len(clips)} clips of {clips[0].shape} "
          f"@ {MINIMAX_H3_FPS} fps ({time.perf_counter() - t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    enc = build_encoder(dtype)
    dec = build_decoder(dtype, enc.vae)
    print(f"model: vae + audio_vae loaded in {time.perf_counter() - t0:.1f}s "
          f"({args.dtype}, {sum(p.numel() for p in enc.vae.parameters()) / 1e6:.0f}M vae params)", flush=True)

    # Encode every clip once, untimed: the latents are the decoder's input.
    t0 = time.perf_counter()
    latents = []
    with torch.no_grad():
        for frames in clips:
            ref = MiniMaxH3VideoReference(frames=frames, fps=float(MINIMAX_H3_FPS))
            out = enc(references=[ref], num_frames=args.frames, output="condition_latents")
            lat = out[0] if isinstance(out, (list, tuple)) else out
            lat = lat[0] if isinstance(lat, list) else lat
            latents.append(lat.to("cuda", dtype))
    torch.cuda.synchronize()
    print(f"encode: {len(latents)} clips -> latents {tuple(latents[0].shape)} "
          f"({time.perf_counter() - t0:.1f}s, untimed)", flush=True)

    rows, walls = [], []
    with torch.no_grad():
        for i, lat in enumerate(latents):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dec(latents=lat, output_type="pt", output="videos")
            video = out[0] if isinstance(out, (list, tuple)) else out
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            walls.append(dt)
            v = video[0] if video.dim() == 5 else video            # (T, C, H, W) or (C, T, H, W)
            vf = v.float().cpu()
            ref = torch.from_numpy(clips[i]).float()
            if ref.max() > 1.5:
                ref = ref / 255.0
            mae = None
            if vf.dim() == 4 and ref.dim() == 4:
                a = vf.permute(0, 2, 3, 1) if vf.shape[1] in (1, 3) else vf
                if a.shape == ref.shape:
                    mae = float((a - ref).abs().mean())
            rows.append({"clip": i, "s": round(dt, 4), "shape": list(video.shape), "mae": mae})
            print(f"  clip {i} {dt:.3f}s -> {tuple(video.shape)}"
                  + (f" mae {mae:.4f}" if mae is not None else ""), flush=True)

    med = statistics.median(walls)
    print(f"decode: median {med:.3f}s per {args.frames}-frame clip over {len(walls)} clips "
          f"({args.frames / med:.1f} frames/s), peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"clips": rows, "median_s": med, "frames": args.frames, "dtype": args.dtype,
                   "peak_gb": torch.cuda.max_memory_allocated() / 1e9}, fh, indent=1)


if __name__ == "__main__":
    main()
