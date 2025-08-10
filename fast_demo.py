# fast_demo.py (fixed)
# Minimal-VRAM end-to-end demo for p2p_edition:
# inversion (DDIM) -> null-text optimization -> single edit

import os
import gc
import argparse
from datetime import datetime

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

# repo-local imports
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..")) if os.path.basename(HERE) == "notebooks" else HERE
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from pipelines.real_image_editor import RealImageEditor
import ptp_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--source", type=str, default="a cat sitting next to a mirror", help="Source prompt")
    p.add_argument("--target", type=str, default="a tiger sitting next to a mirror", help="Target prompt")
    p.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "outputs"), help="Output directory")
    p.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4", help="SD model repo id")
    # Fast-demo knobs (tuned for 8GB)
    p.add_argument("--inv_steps", type=int, default=20, help="DDIM inversion steps")
    p.add_argument("--null_iters", type=int, default=60, help="Null-text optimization iters")
    p.add_argument("--edit_steps", type=int, default=32, help="Editing denoise steps")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale for editing")
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


from PIL import Image
import numpy as np

def to_pil(x):
    """Convert PIL or HxWxC uint8 ndarray to PIL.Image."""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, np.ndarray):
        # ensure uint8 RGB
        arr = x
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(x)}")

def save_grid(images, labels, out_path):
    """Make a 1-row labeled grid (robust to PIL or ndarray inputs)."""
    labeled = [to_pil(ptp_utils.text_under_image(img, lab)) for img, lab in zip(images, labels)]
    widths = [im.width for im in labeled]
    heights = [im.height for im in labeled]
    total_w, max_h = sum(widths), max(heights)
    grid = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in labeled:
        grid.paste(im, (x, 0))
        x += im.width
    grid.save(out_path)

def resize_to_512(in_path):
    """Force 512x512 to keep UNet memory small."""
    img = Image.open(in_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    tmp = os.path.join(os.path.dirname(in_path), "_tmp_demo_512.png")
    img.save(tmp)
    return tmp


def build_pipeline(model_id, device):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        # Optional: get rid of the warning by using leading spacing step offset=1
        steps_offset=1,
        timestep_spacing="leading",
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,  # optional (install accelerate for best effect)
    ).to(device)

    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("xformers not available:", e)
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        if dtype == torch.float16:
            pipe.vae.to(dtype=torch.float16)

    return pipe


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assert os.path.exists(args.image), f"Image not found: {args.image}"
    img_path_512 = resize_to_512(args.image)
    ensure_dir(args.outdir)

    print("Loading model...")
    model = build_pipeline(args.model, device)
    print("✅ Model ready.")

    gc.collect(); torch.cuda.empty_cache()

    editor = RealImageEditor(model, device=device)

    # Try to set inversion steps on inverter if exposed
    try:
        editor.ddim_inverter.num_inference_steps = args.inv_steps
        print(f"Set inverter steps to {args.inv_steps}")
    except Exception:
        print("Inverter does not expose num_inference_steps; using its default.")

    # -------- Phase 1: DDIM inversion --------
    print(f"🔄 DDIM inversion ({args.inv_steps} steps)...")
    gc.collect(); torch.cuda.empty_cache()
    inversion_results = editor.invert_image(
        image_path=img_path_512,
        prompt=args.source,
        offsets=(0, 0, 0, 0),
        cache_key="fast_demo",
    )
    print("✅ Inversion done.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inv_out = os.path.join(args.outdir, f"fast_demo_inversion_{stamp}.png")
    save_grid(
        [inversion_results["original_image"], inversion_results["reconstructed_image"]],
        ["Original", "DDIM"],
        inv_out,
    )
    print("Saved:", inv_out)

    # -------- Phase 2: Null-text optimization --------
    print(f"🎯 Null-text optimization ({args.null_iters} iters)...")
    gc.collect(); torch.cuda.empty_cache()
    optimization_results = editor.optimize_null_text(
        inversion_results=inversion_results,
        num_iterations=args.null_iters,
        cache_key="fast_demo",
    )
    print("✅ Null-text optimization done.")

    nto_out = os.path.join(args.outdir, f"fast_demo_nto_{stamp}.png")
    save_grid(
        [
            inversion_results["original_image"],
            inversion_results["reconstructed_image"],
            optimization_results["reconstructed_image"],
        ],
        ["Original", "DDIM", "DDIM + Null"],
        nto_out,
    )
    print("Saved:", nto_out)

    # -------- Phase 3: Editing --------
    print(f"🎨 Editing: '{args.target}'  (steps={args.edit_steps}, cfg={args.guidance})")
    gc.collect(); torch.cuda.empty_cache()
    editing_results = editor.edit_image(
        inversion_results=inversion_results,
        optimization_results=optimization_results,
        target_prompt=args.target,
        edit_type="replace",
        use_adaptive_scheduling=False,
        manual_params={
            "cross_replace_steps": {"default_": 0.70},
            "self_replace_steps": 0.50,
            "recommended_guidance_scale": args.guidance,
            "recommended_num_inference_steps": args.edit_steps,
        },
    )
    print("✅ Edit done.")

    edit_out = os.path.join(args.outdir, f"fast_demo_edit_{stamp}.png")
    save_grid(
        [inversion_results["original_image"], editing_results["edited_images"][1]],
        ["Original", "Edited"],
        edit_out,
    )
    print("Saved:", edit_out)

    print("\nAll done.")
    print("Outputs:")
    print(" -", inv_out)
    print(" -", nto_out)
    print(" -", edit_out)


if __name__ == "__main__":
    main()
