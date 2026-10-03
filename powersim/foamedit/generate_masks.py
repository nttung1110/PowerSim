"""2D object masks for a scene's photos from text prompts (Grounding DINO box + SAM mask).

    python -m powersim.foamedit.generate_masks --images-dir <photos> --out-root <dir> \
        --prompt vase="a vase." --prompt frond="a dried palm frond."

Writes <out-root>/<name>/<photo stem>.png, the layout powersim.foamedit.select reads."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_models(device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device).eval()
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    return gd_processor, gd_model, sam_processor, sam_model


def mask_for_prompt(image, prompt, models, device, box_threshold=0.3, text_threshold=0.25):
    """(mask (H, W) bool, detection score) for the best box of `prompt`, or (None, None)"""
    gd_processor, gd_model, sam_processor, sam_model = models
    gd_inputs = gd_processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gd_outputs = gd_model(**gd_inputs)
    results = gd_processor.post_process_grounded_object_detection(
        gd_outputs, gd_inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    if len(results["boxes"]) == 0:
        return None, None
    best = results["scores"].argmax().item()
    box = results["boxes"][best].tolist()
    score = results["scores"][best].item()
    sam_inputs = sam_processor(image, input_boxes=[[box]], return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)
    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
    )[0]
    best_mask = sam_outputs.iou_scores.cpu()[0, 0].argmax().item()
    return masks[0, best_mask].numpy().astype(bool), score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, help="the scene's training photos (any resolution)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prompt", action="append", required=True, metavar="NAME=TEXT",
                    help="repeatable; e.g. --prompt vase='a vase.' (end prompts with a period)")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    prompts = {}
    for entry in a.prompt:
        name, sep, text = entry.partition("=")
        if not sep:
            ap.error(f"--prompt entries are NAME=TEXT, got {entry!r}")
        prompts[name] = text

    models = load_models(a.device)
    images = sorted(f for f in os.listdir(a.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"[generate_masks] {len(images)} photos in {a.images_dir}; prompts {prompts}", flush=True)
    out_root = Path(a.out_root)
    for name in prompts:
        (out_root / name).mkdir(parents=True, exist_ok=True)
    missing = {k: 0 for k in prompts}
    low = {k: 0 for k in prompts}
    t0 = time.time()
    for i, fname in enumerate(images):
        image = Image.open(os.path.join(a.images_dir, fname)).convert("RGB")
        stem = os.path.splitext(fname)[0]
        for name, text in prompts.items():
            mask, score = mask_for_prompt(image, text, models, a.device, a.box_threshold, a.text_threshold)
            if mask is None:
                missing[name] += 1
                mask = np.zeros((image.height, image.width), dtype=bool)
            elif score < 0.5:
                low[name] += 1
            Image.fromarray((mask.astype(np.uint8) * 255)).save(out_root / name / f"{stem}.png")
        if (i + 1) % 20 == 0 or i + 1 == len(images):
            print(f"[generate_masks] {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)
    for name in prompts:
        print(f"[generate_masks] {name}: detected in {len(images) - missing[name]}/{len(images)} photos "
              f"({low[name]} with box score < 0.5) -> {out_root / name}", flush=True)


if __name__ == "__main__":
    main()
