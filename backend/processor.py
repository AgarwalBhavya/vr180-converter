# import os, sys, uuid, json, time
# import numpy as np, cv2
# import torch
# from torchvision import transforms
# from status_store import write_status
# from utils import extract_frames, encode_side_by_side
#
# # ---------- config ----------
# BASELINE_PX = 28         # tune this: larger = stronger parallax
# SMOOTH_ALPHA = 0.6       # blending weight for temporal depth smoothing
# TMP_DIR = None           # set per-job
#
# # ---------- MiDaS load ----------
# def load_midas():
#     # MiDaS small for speed
#     model_type = "MiDaS_small"
#     model = torch.hub.load("intel-isl/MiDaS", model_type)
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
#     midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#     transform = midas_transforms.small_transform
#     return model, transform
#
# # ---------- helpers ----------
# def update_status(job_dir, stage, progress=0.0, preview=None, download=None):
#     obj = {'stage': stage, 'progress': progress, 'preview': preview, 'download': download}
#     write_status(job_dir, obj)
#
# def read_images_sorted(frames_dir):
#     files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
#     return [os.path.join(frames_dir, f) for f in files]
#
# def estimate_depth_for_frame(img_bgr, model, transform, device):
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#
#     # pass raw image to transform
#     inp = transform(img_rgb)
#
#     # Ensure correct shape: [1, 3, H, W]
#     if inp.dim() == 3:  # [3, H, W]
#         inp = inp.unsqueeze(0)
#     inp = inp.to(device)
#
#     with torch.no_grad():
#         pred = model(inp)
#         pred = torch.nn.functional.interpolate(
#             pred.unsqueeze(1),
#             size=img_rgb.shape[:2],
#             mode="bicubic",
#             align_corners=False
#         ).squeeze().cpu().numpy()
#
#     # normalize
#     pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
#     return pred
#
# import numpy as np
# import cv2
#
# def generate_stereo_pair(img_bgr, depth, disparity_scale=30):
#     """
#     Generate left and right eye images using depth-based disparity.
#     """
#     h, w, _ = img_bgr.shape
#     depth_resized = cv2.resize(depth, (w, h))
#
#     # Normalize disparity shift
#     disp = (depth_resized - 0.5) * disparity_scale
#
#     # Create x-coordinates grid
#     x_coords = np.tile(np.arange(w), (h, 1)).astype(np.float32)
#
#     # Left and right maps
#     map_left = (x_coords - disp).clip(0, w-1)
#     map_right = (x_coords + disp).clip(0, w-1)
#
#     # Remap using disparity
#     left_img = cv2.remap(img_bgr, map_left, np.tile(np.arange(h)[:, None], (1, w)).astype(np.float32),
#                          interpolation=cv2.INTER_LINEAR)
#     right_img = cv2.remap(img_bgr, map_right, np.tile(np.arange(h)[:, None], (1, w)).astype(np.float32),
#                           interpolation=cv2.INTER_LINEAR)
#
#     return left_img, right_img
#
#
# def process_video(input_path, output_path, model, transform, device):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = None
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # --- Add your depth + stereo pair code here ---
#         depth = estimate_depth_for_frame(frame, model, transform, device)
#         left_img, right_img = generate_stereo_pair(frame, depth)
#
#         # concatenate side by side
#         sbs = np.concatenate([left_img, right_img], axis=1)
#
#         if out is None:
#             h, w, _ = sbs.shape
#             out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
#
#         out.write(sbs)
#
#     cap.release()
#     out.release()
#
#
#
# def warp_by_disparity(frame, depth, baseline, side='right'):
#     h,w = depth.shape
#     # disparity larger for closer objects
#     disparity = (baseline * (1.0 - depth)).astype(np.float32)
#     if side == 'left':
#         disparity = -disparity
#     xs, ys = np.meshgrid(np.arange(w), np.arange(h))
#     map_x = (xs - disparity).astype(np.float32)
#     map_y = ys.astype(np.float32)
#     warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#     return warped
#
# def inpaint_masked(img_bgr, mask):
#     # try LaMa if available (user can integrate) - fallback to OpenCV
#     try:
#         # placeholder for LaMa call; user can hook here
#         raise Exception("LaMa not hooked")
#     except Exception:
#         # OpenCV inpaint expects 8-bit mask
#         mask_u8 = (mask.astype('uint8') * 255)
#         inpainted = cv2.inpaint(img_bgr, mask_u8, 3, cv2.INPAINT_TELEA)
#         return inpainted
#
# # Temporal warp helper using optical flow (Farneback)
# def warp_image_with_flow(img, flow):
#     h,w = flow.shape[:2]
#     xs, ys = np.meshgrid(np.arange(w), np.arange(h))
#     map_x = (xs + flow[...,0]).astype(np.float32)
#     map_y = (ys + flow[...,1]).astype(np.float32)
#     return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#
# def compute_flow(prev_gray, curr_gray):
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
#                                         None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     return flow
#
# # ---------- pipeline entry ----------
# def run_pipeline(job_dir, input_video):
#     start = time.time()
#     global TMP_DIR
#     TMP_DIR = os.path.join(job_dir, 'work')
#     os.makedirs(TMP_DIR, exist_ok=True)
#     frames_dir = os.path.join(TMP_DIR, 'frames')
#     out_left = os.path.join(TMP_DIR, 'out_left')
#     out_right = os.path.join(TMP_DIR, 'out_right')
#     os.makedirs(out_left, exist_ok=True)
#     os.makedirs(out_right, exist_ok=True)
#
#     update_status(job_dir, 'extracting_frames', 0.05)
#     rc = extract_frames(input_video, frames_dir)
#     if rc != 0:
#         update_status(job_dir, 'failed', 0.0)
#         return
#
#     frame_paths = read_images_sorted(frames_dir)
#     if len(frame_paths) == 0:
#         update_status(job_dir, 'failed', 0.0)
#         return
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model, transform = load_midas()
#     prev_depth = None
#     prev_gray = None
#
#     total = len(frame_paths)
#     for idx, fp in enumerate(frame_paths):
#         stage = f'processing_frames ({idx+1}/{total})'
#         update_status(job_dir, stage, 0.05 + 0.8 * ((idx)/total))
#         img = cv2.imread(fp)
#         h,w = img.shape[:2]
#
#         # 1. estimate depth
#         depth = estimate_depth_for_frame(img, model, transform, device)
#
#         # 2. temporal smoothing: warp prev depth -> current via flow and blend
#         if prev_depth is not None:
#             curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             flow = compute_flow(prev_gray, curr_gray)
#             warped_prev_depth = warp_image_with_flow(prev_depth, flow)
#             depth = SMOOTH_ALPHA * depth + (1.0 - SMOOTH_ALPHA) * warped_prev_depth
#             prev_gray = curr_gray
#         else:
#             prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         prev_depth = depth.copy()
#
#         # 3. create right-eye by depth warping
#         right = warp_by_disparity(img, depth, BASELINE_PX, side='right')
#
#         # 4. hole mask & inpaint
#         gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
#         mask = (gray_r == 0).astype(np.uint8)
#         if mask.sum() > 0:
#             right = inpaint_masked(right, mask)
#
#         # 5. save outputs (left = original to preserve quality)
#         left_out = os.path.join(out_left, os.path.basename(fp))
#         right_out = os.path.join(out_right, os.path.basename(fp))
#         cv2.imwrite(left_out, img)
#         cv2.imwrite(right_out, right)
#
#         # occasionally write preview frame for UI
#         if idx == min(3, total-1):
#             preview_path = os.path.join(job_dir, 'preview.mp4')
#             # quick preview generation of current frame only
#             cv2.imwrite(os.path.join(job_dir, 'preview_frame.png'), right)
#
#     # 6. encode side-by-side MP4
#     update_status(job_dir, 'encoding', 0.95)
#     left_pattern = os.path.join(out_left, 'frame_%06d.png')
#     right_pattern = os.path.join(out_right, 'frame_%06d.png')
#     final_out = os.path.join(job_dir, 'vr180_sbs.mp4')
#     encode_side_by_side(left_pattern, right_pattern, final_out, fps=24)
#
#     # Write final status with download link (relative path usable from server)
#     download_url = f'/outputs/{os.path.basename(job_dir)}/vr180_sbs.mp4'
#     preview_url = f'/outputs/{os.path.basename(job_dir)}/preview_frame.png' if os.path.exists(os.path.join(job_dir,'preview_frame.png')) else None
#     update_status(job_dir, 'done', 1.0, preview=preview_url, download=download_url)
#     end = time.time()
#     print('job done in', end-start)


# backend/processor.py
import os
import sys
import time
import uuid
import subprocess
from pathlib import Path

import numpy as np
import cv2
import torch

from status_store import write_status
from utils import extract_frames  # keep using your utils.extract_frames

# ---------- config (tweak these) ----------
BASELINE_PX = 48         # total pixel disparity magnitude (tune for your resolution)
SMOOTH_ALPHA = 0.6       # temporal smoothing alpha (0..1)
TMP_SUBDIR = "work"      # job work subfolder
ENCODE_CRF = 18          # quality: lower = better, 18 is near visually lossless
ENCODE_PRESET = "slow"   # ffmpeg x264 preset
# ------------------------------------------

def update_status(job_dir, stage, progress=0.0, preview=None, download=None):
    obj = {'stage': stage, 'progress': progress, 'preview': preview, 'download': download}
    write_status(job_dir, obj)


# ---------- MiDaS loader ----------
def load_midas(device=None):
    """
    Load MiDaS small for speed. Returns (model, transform)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_type = "MiDaS_small"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    return model, transform


# ---------- depth estimation ----------
def estimate_depth_for_frame(img_bgr, model, transform, device):
    """
    img_bgr: uint8 BGR image (H,W,3)
    returns: float32 depth map normalized 0..1 (H,W)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # MiDaS transform expects raw numpy image for small_transform
    inp = transform(img_rgb)

    # Make sure shape is [1,3,H,W]
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    inp = inp.to(device)

    with torch.no_grad():
        pred = model(inp)
        # pred shape -> [batch, H', W'] maybe; interpolate to original size
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize to 0..1
    pred = pred.astype(np.float32)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred


# ---------- stereo warping helpers ----------
def _coord_maps(h, w):
    xs = np.repeat(np.arange(w, dtype=np.float32)[None, :], h, axis=0)
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    return xs, ys


def warp_with_disparity(img_bgr, disp):
    """
    Warp image horizontally by disp (HxW float32) where positive disp shifts source right.
    Returns warped_img (HxWx3 uint8) and valid mask (HxW uint8: 1 valid, 0 hole)
    """
    h, w = disp.shape
    xs, ys = _coord_maps(h, w)
    map_x = (xs - disp).astype(np.float32)  # dest->src mapping
    map_y = ys.astype(np.float32)
    warped = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_mask = (warped.sum(axis=2) != 0).astype(np.uint8)
    return warped, valid_mask


def fill_holes_inpaint(img_bgr, mask):
    """
    Fill holes (mask==0) using OpenCV inpaint (Telea).
    mask: 1 valid, 0 hole.
    """
    holes = (1 - mask).astype(np.uint8) * 255  # inpaint mask: 255 = fill
    if holes.sum() == 0:
        return img_bgr
    # ensure 8-bit
    img8 = img_bgr if img_bgr.dtype == np.uint8 else np.clip(img_bgr, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(img8, holes, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # blend at edges: dilate hole mask and mix
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(holes, kernel, iterations=1) - holes
    if edge.sum() > 0:
        edge_f = (edge.astype(np.float32) / 255.0)[:, :, None]
        out = img8.astype(np.float32) * (1.0 - edge_f) + filled.astype(np.float32) * edge_f
        return np.clip(out, 0, 255).astype(np.uint8)
    return filled


def generate_stereo_images(img_bgr, depth_norm, baseline_px=BASELINE_PX, invert_depth=False):
    """
    Generate left and right images from a single input using the depth map.
    - baseline_px: total separation (pixels). We use +/- baseline/2 per-eye for symmetry.
    - invert_depth: toggle if nearer/farther mapping is reversed for your MiDaS variant.
    Returns (left_bgr, right_bgr) both uint8.
    """
    h, w = img_bgr.shape[:2]
    depth = cv2.resize(depth_norm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if invert_depth:
        depth = 1.0 - depth

    # convert depth (0..1) -> disparity in pixels. Center depth -> zero disp
    # shift = (depth - 0.5) * baseline_px
    shift = (depth - 0.5) * baseline_px

    # half baseline per eye (center the stereo around original)
    left_disp = -0.5 * shift  # left eye samples from slightly right
    right_disp = +0.5 * shift

    left_warp, left_mask = warp_with_disparity(img_bgr, left_disp)
    right_warp, right_mask = warp_with_disparity(img_bgr, right_disp)

    left_filled = fill_holes_inpaint(left_warp, left_mask)
    right_filled = fill_holes_inpaint(right_warp, right_mask)

    # small horizontal smoothing to reduce seam artifacts
    left_out = cv2.GaussianBlur(left_filled, (3, 1), 0)
    right_out = cv2.GaussianBlur(right_filled, (3, 1), 0)
    return left_out, right_out


# ---------- encoding helper ----------
def encode_side_by_side(left_pattern, right_pattern, out_path, fps=24):
    """
    left_pattern and right_pattern are format strings like /path/frame_%06d.png
    Encodes them into a side-by-side mp4 using ffmpeg hstack filter.
    """
    # ffmpeg command: provide both sequences with matching frame numbering
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", left_pattern,
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", right_pattern,
        "-filter_complex", "hstack=inputs=2",
        "-c:v", "libx264",
        "-preset", ENCODE_PRESET,
        "-crf", str(ENCODE_CRF),
        "-pix_fmt", "yuv420p",
        out_path
    ]
    # run and raise on error
    subprocess.check_call(cmd)


# ---------- pipeline ----------
def run_pipeline(job_dir, input_video):
    start_t = time.time()
    job_dir = str(job_dir)
    work_dir = os.path.join(job_dir, TMP_SUBDIR)
    frames_dir = os.path.join(work_dir, "frames")
    left_dir = os.path.join(work_dir, "left")
    right_dir = os.path.join(work_dir, "right")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    try:
        update_status(job_dir, "extracting_frames", 0.02)
        rc = extract_frames(input_video, frames_dir)  # should produce frame_000000.png ... or frame_000001.png
        if rc != 0:
            update_status(job_dir, "failed_extract", 0.0)
            return

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        total = len(frame_files)
        if total == 0:
            update_status(job_dir, "no_frames", 0.0)
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, transform = load_midas(device=device)

        prev_depth = None
        prev_gray = None

        # try to detect start_number used by extract_frames, check first filename
        first_name = frame_files[0]
        # decide start_number base (0 or 1)
        if first_name.endswith("000001.png") or first_name.endswith("000001.jpg"):
            start_number = 1
        else:
            start_number = 0

        for idx, fname in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, fname)
            update_status(job_dir, f'processing_frames ({idx+1}/{total})', 0.02 + 0.8 * (idx / max(1, total)))
            img = cv2.imread(frame_path)
            if img is None:
                continue

            # depth estimation
            depth = estimate_depth_for_frame(img, model, transform, device)

            # temporal smoothing via optical flow (optional)
            if prev_depth is not None:
                curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # warp prev depth to current frame using flow (flow is dx,dy)
                h, w = depth.shape
                xs, ys = _coord_maps(h, w)
                map_x = (xs + flow[..., 0]).astype(np.float32)
                map_y = (ys + flow[..., 1]).astype(np.float32)
                warped_prev = cv2.remap(prev_depth, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
                depth = SMOOTH_ALPHA * depth + (1.0 - SMOOTH_ALPHA) * warped_prev
                prev_gray = curr_gray
            else:
                prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            prev_depth = depth.copy()

            # generate stereo (both eyes)
            left_img, right_img = generate_stereo_images(img, depth, baseline_px=BASELINE_PX, invert_depth=False)

            # save frames (use same filename)
            cv2.imwrite(os.path.join(left_dir, fname), left_img)
            cv2.imwrite(os.path.join(right_dir, fname), right_img)

            # write a small preview frame early (SBS)
            if idx == min(3, total - 1):
                sbs_preview = np.concatenate([cv2.resize(left_img, (left_img.shape[1]//2, left_img.shape[0]//2)),
                                              cv2.resize(right_img, (right_img.shape[1]//2, right_img.shape[0]//2))], axis=1)
                preview_png = os.path.join(job_dir, "preview_frame.png")
                cv2.imwrite(preview_png, sbs_preview)

        update_status(job_dir, "encoding", 0.90)

        # prepare patterns for ffmpeg. Ensure numbering matches extract_frames start
        # assume frame filenames are like frame_000000.png or frame_000001.png
        # create pattern with same padding
        example = frame_files[0]
        left_pattern = os.path.join(left_dir, example.replace(example[-10:], "%06d.png")) if example.endswith(".png") else os.path.join(left_dir, "frame_%06d.png")
        # fallback: simpler pattern
        left_pattern = os.path.join(left_dir, "frame_%06d.png")
        right_pattern = os.path.join(right_dir, "frame_%06d.png")

        final_out = os.path.join(job_dir, "vr180_sbs.mp4")
        # try to get fps from video if available; else default 24
        fps = 24
        try:
            # if the input video still exists as file, probe fps
            cap = cv2.VideoCapture(input_video)
            f = cap.get(cv2.CAP_PROP_FPS)
            if f and f > 0:
                fps = f
            cap.release()
        except Exception:
            pass

        encode_side_by_side(left_pattern, right_pattern, final_out, fps=fps)

        download_url = f'/outputs/{os.path.basename(job_dir)}/vr180_sbs.mp4'
        preview_url = f'/outputs/{os.path.basename(job_dir)}/preview_frame.png' if os.path.exists(os.path.join(job_dir, 'preview_frame.png')) else None
        update_status(job_dir, 'done', 1.0, preview=preview_url, download=download_url)

        elapsed = time.time() - start_t
        print(f"Job finished in {elapsed:.1f}s -> {final_out}")

    except Exception as e:
        # log the error for debugging
        import traceback
        traceback.print_exc()
        update_status(job_dir, 'failed', 0.0)

