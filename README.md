# h3-vae-bench

MiniMax-H3 video VAE through the documented diffusers modular pipeline,
loading only the `vae` and `audio_vae` components of `MiniMaxAI/MiniMax-H3`:

- `encode.py` -- a 124-frame reference clip to condition latents
  (`Ref2VA` setup + reference encoder), 3 timed runs.
- `decode.py` -- 4 clips encoded once untimed, then `MiniMaxH3VideoDecodeStep`
  (the 36-layer ViT decoder + pixel un-normalisation) timed once per clip.

```
pip install -r requirements.txt
python encode.py
python decode.py
```
