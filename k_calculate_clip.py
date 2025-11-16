#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List, Tuple, Dict, Optional

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel # type: ignore
import cv2 # type: ignore


class CLIPScoreCalculator:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name) # type: ignore
        self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.to(device=self.device) # type: ignore
        self.model.eval()

    def compute_score(self, frames: List[Image.Image], text: str) -> torch.Tensor:
        """
        Returns per-frame CLIP cosine similarities. Shape: [N]
        """
        # Images batch
        img_batch = self.processor(images=frames, return_tensors="pt", padding=True)
        pixel_values: torch.Tensor = img_batch["pixel_values"].to(self.device) # [N, 3, H, W]
        N: int = int(pixel_values.shape[0])

        # Text batch (compute once)
        txt_batch = self.processor(text=[text], return_tensors="pt", padding=True)
        input_ids: torch.Tensor = txt_batch["input_ids"].to(self.device) # [1, L]
        attention_mask: torch.Tensor = txt_batch["attention_mask"].to(self.device) # [1, L]

        with torch.no_grad():
            image_features: torch.Tensor = self.model.get_image_features(pixel_values=pixel_values) # [N, D]
            text_features: torch.Tensor = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ) # [1, D]
            text_features = text_features.expand(N, -1) # [N, D]

        # Normalize to unit vectors
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [N, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [N, D]

        # Cosine similarity per frame
        sim_scores: torch.Tensor = (image_features * text_features).sum(dim=-1) # [N]
        return sim_scores

    def compute_statistics(self, sim_scores: torch.Tensor) -> Tuple[float, float, Tuple[float, float]]:
        mean_score: float = sim_scores.mean().item()
        std_score: float = sim_scores.std(unbiased=False).item()
        conf95: Tuple[float, float] = (
            mean_score - 1.96 * std_score / len(sim_scores)**0.5,
            mean_score + 1.96 * std_score / len(sim_scores)**0.5,
        )
        return mean_score, std_score, conf95


_VIDEO_EXTS_DEFAULT: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")


def list_videos(root: pathlib.Path, exts: Tuple[str, ...], recursive: bool) -> List[pathlib.Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    videos: List[pathlib.Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                videos.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                videos.append(p)
    videos.sort()
    return videos


def _humanize_stem(stem: str) -> str:
    text: str = re.sub(r"[_\-]+", " ", stem)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sample_indices_uniform(n_total: int, n_samples: int) -> List[int]:
    if n_total <= 0:
        return []
    n: int = max(1, min(n_samples, n_total))
    idx: torch.Tensor = torch.linspace(0, n_total - 1, steps=n, dtype=torch.float32) # [n]
    idx = idx.round().to(torch.int64) # [n]
    unique_idx: torch.Tensor = torch.unique(idx) # [<=n]
    return [int(i.item()) for i in unique_idx]


def _read_frames_at_indices(video_path: pathlib.Path, indices: List[int]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return frames

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

    cap.release()
    return frames


def score_single_video(
    calc: CLIPScoreCalculator,
    video_path: pathlib.Path,
    n_frames: int,
) -> Optional[torch.Tensor]:
    """
    Returns per-frame similarities for this video, or None on failure.
    Shape: [N_video] where N_video == sampled & decoded frames (≈ n_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total_frames_f: float = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    total_frames: int = int(total_frames_f) if total_frames_f and total_frames_f > 0 else 0
    idxs: List[int] = _sample_indices_uniform(total_frames, n_frames)
    frames: List[Image.Image] = _read_frames_at_indices(video_path, idxs)
    if len(frames) == 0:
        return None
    
    if ('/airport/' in str(video_path)) or ("an_airport_tile_with_a_runway_and_control_tower" in str(video_path)):
        text = 'airport'
    elif ('/amusement_park/' in str(video_path)) or ("an_amusement_park_tile_with_a_roller_coaster_and_ferris_wheel" in str(video_path)):
        text = 'amusement park'
    elif ('/ancient_rome/' in str(video_path)) or ("an_ancient_rome_tile_with_forums_and_columned_porticoes" in str(video_path)):
        text = 'ancient rome'
    elif ('/city/' in str(video_path)) or ("a_large_city_block_with_parks" in str(video_path)):
        text = 'city'
    elif ('/college/' in str(video_path)) or ("a_college_campus_tile_with_academic_buildings_and_green_spaces" in str(video_path)):
        text = 'college'
    elif ('/cyberpunk/' in str(video_path)) or ("a_cyberpunk_city_tile_with_a_street_market_and_modular_stalls" in str(video_path)):
        text = 'cyberpunk'
    elif ('/desert/' in str(video_path)) or ("a_desert_tile_with_dune_fields_and_scattered_cacti" in str(video_path)):
        text = 'desert'
    elif ('/forest/' in str(video_path)) or ("a_dense_forest_tile_with_tall_trees_and_a_small_clearing" in str(video_path)):
        text = 'forest'
    elif ('/lego/' in str(video_path)) or ("a_lego_city_tile_with_buildings_and_roads" in str(video_path)):
        text = 'lego'
    elif ('/medieval/' in str(video_path)) or ("a_medieval_tile_with_farmland_fields_and_cottages" in str(video_path)):
        text = 'medieval'
    elif ('/minecraft/' in str(video_path)) or ("a_minecraft_terrain_tile_with_rolling_hills_and_trees" in str(video_path)):
        text = 'minecraft'
    elif ('/ocean/' in str(video_path)) or ("an_ocean_tile_with_open_water_and_a_navigation_buoy" in str(video_path)):
        text = 'ocean'
    elif ('/park/' in str(video_path)) or ("a_park_tile_with_a_playground_and_picnic_area" in str(video_path)):
        text = 'park'
    elif ('/room/' in str(video_path)) or ("a_modern_living_room_tile_with_a_sofa,_coffee_table,_and_TV" in str(video_path)):
        text = 'room'
    elif ('/winter/' in str(video_path)) or ("a_snowy_winter_tile_with_pine_trees_and_a_frozen_lake" in str(video_path)):
        text = 'winter'
    else:
        raise RuntimeError(f"Unknown category for video: {video_path}")
    text = _humanize_stem(text)
    text =  f'{text}, bird eye view'
    print(f"Prompt: '{text}'")
    sim_scores: torch.Tensor = calc.compute_score(frames, text) # [N_video]
    return sim_scores


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CLIP scores for videos using filename as text.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing videos.")
    parser.add_argument("--n-frames", type=int, default=16, help="Uniform samples per video.")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="HF model id for CLIP.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--only-close", action="store_true", help="Only process files with '.close.mp4' in the filename.")
    parser.add_argument("--dry", action="store_true", help="Dry run; list files but do not process.")
    parser.add_argument(
        "--exts",
        type=str,
        default=",".join(_VIDEO_EXTS_DEFAULT),
        help=f"Comma-separated extensions (default: {','.join(_VIDEO_EXTS_DEFAULT)})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root: pathlib.Path = pathlib.Path(args.dir)
    exts: Tuple[str, ...] = tuple([e.strip().lower() if e.startswith(".") else f".{e.strip().lower()}" for e in args.exts.split(",") if e.strip()])

    videos: List[pathlib.Path] = list_videos(root, exts, recursive=bool(args.recursive))
    videos = [v for v in videos if (not args.only_close) or ('.close' in v.name)]
    if len(videos) == 0:
        print(f"[WARN] No videos found in {root} with extensions: {exts}")
        return 1

    print(f"[INFO] Found {len(videos)} video(s). Initializing CLIP…", flush=True)
    calc = CLIPScoreCalculator(model_name=args.model)
    
    if args.dry:
        for v in videos:
            print(f"[DRY] Would process: {v}")
        return 0

    per_video_frame_scores: Dict[str, torch.Tensor] = {}
    n_ok: int = 0
    n_fail: int = 0

    for v in videos:
        sim_scores: Optional[torch.Tensor] = score_single_video(calc, v, n_frames=int(args.n_frames))
        if sim_scores is None or sim_scores.numel() == 0:
            print(f"[FAIL] {str(v)}: could not decode or no frames.", flush=True)
            n_fail += 1
            continue
        per_video_frame_scores[str(v)] = sim_scores.cpu() # store on CPU; shape [N_v]
        n_ok += 1
        print(f"[OK]   {str(v)}: mean={sim_scores.mean().item():.4f} (N={sim_scores.numel()})", flush=True)

    if n_ok == 0:
        print("[ERROR] No successful videos; aborting.")
        return 2

    # Per-video reporting (mean across frames within video)
    print("\n=== Per-video mean scores ===")
    width_name: int = max(len(k) for k in per_video_frame_scores.keys())
    for name, s in sorted(per_video_frame_scores.items(), key=lambda kv: kv[0].lower()):
        mean_i: float = float(s.mean().item())
        std_i: float = float(s.std(unbiased=False).item()) if s.numel() > 1 else 0.0
        print(f"{name:<{width_name}}  mean={mean_i:.6f}  std={std_i:.6f}  N_frames={s.numel()}")

    # Overall statistics across ALL FRAMES (not per-video means)
    all_frames_scores: torch.Tensor = torch.cat(list(per_video_frame_scores.values()), dim=0) # [sum_i N_i]
    mean_score, std_score, (ci_lo, ci_hi) = calc.compute_statistics(all_frames_scores)

    expected_N: int = n_ok * int(args.n_frames) # target frames if all decoded
    actual_N: int = int(all_frames_scores.numel())

    print("\n=== Overall statistics (across all sampled frames) ===")
    print(f"videos_ok    = {n_ok} / {len(videos)}  (failed: {n_fail})")
    print(f"N_expected   = {expected_N}  # n_frames_per_video * n_videos_ok")
    print(f"N_actual     = {actual_N}   # sum of decoded frames over all videos")
    print(f"mean         = {mean_score:.6f}")
    print(f"std          = {std_score:.6f}")
    print(f"95% CI       = [{ci_lo:.6f}, {ci_hi:.6f}]")

    if n_fail > 0 or actual_N != expected_N:
        print("\n[NOTE] Some videos yielded fewer frames than requested; statistics use N_actual.")
        
    # store everything printed in a log file
    log_path: pathlib.Path = root / "clip_scores.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Directory: {root}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Videos found: {len(videos)}\n")
        f.write(f"Videos processed (ok/fail): {n_ok}/{n_fail}\n")
        f.write(f"N_expected (if all decoded): {expected_N}\n")
        f.write(f"N_actual (decoded frames): {actual_N}\n")
        f.write(f"\n=== Per-video mean scores ===\n")
        for name, s in sorted(per_video_frame_scores.items(), key=lambda kv: kv[0].lower()):
            mean_i: float = float(s.mean().item())
            std_i: float = float(s.std(unbiased=False).item()) if s.numel() > 1 else 0.0
            f.write(f"{name:<{width_name}}  mean={mean_i:.6f}  std={std_i:.6f}  N_frames={s.numel()}\n")
        f.write("\n=== Overall statistics (across all sampled frames) ===\n")
        f.write(f"videos_ok    = {n_ok} / {len(videos)}  (failed: {n_fail})\n")
        f.write(f"N_expected   = {expected_N}  # n_frames_per_video * n_videos_ok\n")
        f.write(f"N_actual     = {actual_N}   # sum of decoded frames over all videos\n")
        f.write(f"mean         = {mean_score:.6f}\n")
        f.write(f"std          = {std_score:.6f}\n")
        f.write(f"95% CI       = [{ci_lo:.6f}, {ci_hi:.6f}]\n")
        if n_fail > 0 or actual_N != expected_N:
            f.write("\n[NOTE] Some videos yielded fewer frames than requested; statistics use N_actual.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
