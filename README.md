# 🌌 Gsplat_AgileX — 3D Gaussian Splatting Pipeline

> Full end-to-end pipeline for reconstructing 3D scenes from 360° Insta360 footage using Gaussian Splatting. Built for Ubuntu + NVIDIA GPU environments.

---

## ⚡ Prerequisites

Before anything else, set up your environment by following the full installation guide:
👉 [Gsplat Installation Tutorial](https://smartdatascan.com/tutorials/gaussian-splatting-windows/installation/gsplat/)

**Key requirements:**
- **Linux only** — Ubuntu recommended. Gsplat has broken dependencies on Windows.
- **Secure boot must be OFF** — required for NVIDIA driver compatibility.
- **NVIDIA GPU** — CUDA is essential for training.

---

## 🚀 Quick Start

Once everything is installed, activate your environment every session:

```bash
conda activate gsplat
```

---

## 🎥 Step 1 — Capture Footage

Record your scene with the **Insta360 camera**. Keep the clip **under 45–60 seconds** — the fisheye lens captures significantly more data than a standard camera, so frame counts explode fast.

---

## 💻 Step 2 — Export on Windows (One-Time Only)

Use **Insta360 Studio** (Windows only) to export the raw footage:

- Export as a **360 Video**
- Use the **highest bitrate** available
- Output will be **1GB+** — that's expected

Then transfer the `.mp4` to your Ubuntu machine (email, USB, etc.).

---

## 🖼️ Step 3 — Extract Frames with FFmpeg

Extract 1 frame per second from the raw 360 video:

```bash
ffmpeg -i ~/Documents/gsplat_project/Raw_Videos/desk_acfr_robot.mp4 \
  -map 0:0 -vf "fps=1" \
  ~/Documents/gsplat_project/FFMPEG_processed/desk_acfr_robot/front_%04d.jpg
```

> Adjust `fps=1` if you need denser coverage. For fast-moving scenes, try `fps=2` or `fps=3`.

---

## 🔪 Step 4 — Split 360° Frames with AliceVision (Meshroom)

COLMAP can't handle raw equirectangular/fisheye images directly, so we use **AliceVision** to split each frame into 8 perspective crops.

Download **Meshroom 2021.1.0** (specifically the 2021 version):
👉 [Meshroom 2021 on FossHub](https://www.fosshub.com/Meshroom-old.html?dwl=Meshroom-2021.1.0-win64.zip)
📺 [Setup tutorial](https://www.youtube.com/watch?v=LQNBTvgljAw)

### Fix shared library error (Ubuntu)

If you see:
```
error while loading shared libraries: libaliceVision_image.so.2: No such file or directory
```

Run this first:
```bash
export LD_LIBRARY_PATH=~/Downloads/Meshroom-2021.1.0-linux-cuda10/Meshroom-2021.1.0-av2.4.0-centos7-cuda10.2/aliceVision/lib:$LD_LIBRARY_PATH
```

### Run the splitter

Navigate to the AliceVision binary directory:
```bash
cd ~/Downloads/Meshroom-2021.1.0-linux-cuda10/Meshroom-2021.1.0-av2.4.0-centos7-cuda10.2/aliceVision/bin
```

Then run:
```bash
./aliceVision_utils_split360Images \
  -i ~/Documents/gsplat_project/FFMPEG_processed/desk_acfr_robot \
  -o ~/Documents/gsplat_project/AliceVision_processed/desk_acfr_robot \
  --equirectangularNbSplits 8 \
  --equirectangularSplitResolution 1200
```

---

## 🧹 Step 5 — Remove Frames with People (YOLO)

People in the scene = noise in your reconstruction. We use **YOLOv8** to automatically detect and delete frames where a person is visible.

> ⚠️ Before running, open `remove_me.py` and update the `image_folder` path to match your scene directory.

```bash
python ~/Documents/gsplat_project/remove_me.py
```

Install dependencies if needed:
```bash
pip install ultralytics opencv-python
```

---

## 🗺️ Step 6 — Run COLMAP via Nerfstudio

Process the cleaned frames through COLMAP to extract camera poses:

```bash
ns-process-data images \
  --data ~/Documents/gsplat_project/AliceVision_processed/desk_acfr_robot \
  --output-dir ~/Documents/gsplat_project/COLMAP_processed/desk_acfr_robot \
  --camera-type fisheye
```

---

## 🔥 Step 7 — Train the Gaussian Splat

Launch training with `splatfacto`:

```bash
ns-train splatfacto \
  --data ~/Documents/gsplat_project/COLMAP_processed/desk_acfr_robot \
  --pipeline.model.num-random 25000 \
  --pipeline.model.max-gauss-ratio 5.0 \
  --pipeline.datamanager.images-on-gpu False
```

Grab a coffee — this takes a while depending on your GPU. ☕

---

## 🗂️ Pipeline Summary

```
Insta360 footage
      ↓
  FFmpeg (extract frames)
      ↓
  AliceVision (split 360° → perspective crops)
      ↓
  YOLO (remove frames with people)
      ↓
  COLMAP via Nerfstudio (camera pose estimation)
      ↓
  splatfacto (Gaussian Splat training) 🎉
```

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `libaliceVision_image.so.2` not found | Export `LD_LIBRARY_PATH` (see Step 4) |
| NVIDIA driver not loading | Disable Secure Boot in BIOS |
| `gsplat` install fails on Windows | Use Ubuntu — it's not supported |
| YOLO not finding images | Use `os.path.expanduser()` for `~` paths |
| Low reconstruction quality | Increase `fps` in FFmpeg, or increase `--equirectangularSplitResolution` |
