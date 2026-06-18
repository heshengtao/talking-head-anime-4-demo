# THA4 Student Model Training & ONNX Export — Agent Instructions

> Send this file's raw URL to your AI agent (opencode, Claude Code, etc.) and it will guide you through the entire pipeline.

---

## Task Overview

Train a lightweight student model from a single anime character image, then export to a deployable **single-input** ONNX model:

```
output/model.onnx
```

> **Single `pose` input, texture baked in.** No separate files, no runtime preprocessing. Just pass a 45-dim pose vector and receive the rendered green-screen RGB frame.
>
> **Green screen (`#00FF00`) background.** Frontends (Super Agent Party, etc.) chroma-key the green to restore transparency. Dark clothing is not affected.
>
> **`output/model.onnx` is the final artifact.** Import it directly into Super Agent Party or your own app.
>
> Ignore `export_onnx.py` and `merge_onnx.py` — they produce alternative formats not needed for production use.

---

## Prerequisites Check

Before starting, verify:
- [ ] NVIDIA GPU (RTX 2080 or newer)
- [ ] Python 3.10.x installed
- [ ] 30 GB free disk space
- [ ] Git installed

---

## Step 1: Clone & Install Environment

```bash
git clone https://github.com/heshengtao/talking-head-anime-4-demo.git
cd talking-head-anime-4-demo
```

Create Python virtual environment:

```bash
python -m venv venv --prompt talking-head-anime-4-demo
```

Activate and install Poetry:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install Poetry if missing
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -  # Windows
# or: curl -sSL https://install.python-poetry.org | python3 -                                   # Linux/Mac
```

Install project dependencies via Poetry:

```bash
cd poetry
poetry install
cd ..
```

---

## Step 2: Download Required Data Files

### 2.1 THA4 Base Models (~200 MB)

Download URL:
```
https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1
```

Extract to `data/tha4/`. Expected files after extraction:
```
data/tha4/body_morpher.pt
data/tha4/eyebrow_decomposer.pt
data/tha4/eyebrow_morphing_combiner.pt
data/tha4/face_morpher.pt
data/tha4/upscaler.pt
```

### 2.2 Pose Dataset (~200 MB)

Download URL:
```
https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1
```

Save to `data/pose_dataset.pt`.

---

## Step 3: Prepare Character Image

**Constraints (MUST follow):**
- 512×512 pixels, RGBA PNG format
- Alpha channel: background pixels must have alpha = 0 (transparent)
- Only one humanoid character, standing upright, facing forward
- Hands must be below and far from head
- Head should fit in a 128×128 box at the middle of the top half of the image

Place your character image at: `data/images/my_char.png`

---

## Step 4: Create Face Mask with GIMP

Install [GIMP](https://www.gimp.org/) (free, open-source).

Steps to create mask:
1. Open `data/images/my_char.png` in GIMP
2. Add a new layer, fill it with WHITE (255,255,255)
3. Select black color, use paintbrush to paint over eyes and mouth areas
4. Export as PNG: `data/images/my_char_face_mask.png` (512×512, RGB, no alpha channel)

Mask color meaning:
- White = areas the model CAN modify
- Black = eyes and mouth (protected regions)

Reference files: `data/images/lambda_00_face_mask.png`, `data/images/lambda_00_face_mask.xcf`

---

## Step 5: Create Training Configuration

Create directory and config file:

```bash
mkdir data\distill_examples\my_char
```

Write to `data/distill_examples/my_char/config.yaml`:

```yaml
prefix: data/distill_examples/my_char
character_image_file_name: data/images/my_char.png
face_mask_image_file_name: data/images/my_char_face_mask.png
face_morpher_random_seed_0: 12771885812175595441
face_morpher_random_seed_1: 14367217090963479175
body_morpher_random_seed_0: 2892221210020292507
body_morpher_random_seed_1: 9998918537095922080
num_cpu_workers: 1
num_gpus: 1
```

---

## Step 6: Train the Student Model

Training has two stages. Run them sequentially.

### Stage 1 — Face Morpher

~1,000,000 examples, ~3 hours on RTX 4080.

```bash
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_face_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

Wait for this to complete before starting Stage 2.

### Stage 2 — Body Morpher

~1,500,000 examples, ~5 hours on RTX 4080.

```bash
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_body_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

**Monitoring:** Open a separate terminal and run:
```bash
venv\Scripts\python.exe -m tensorboard.main --logdir=data/distill_examples/my_char
```
Then open `http://localhost:6006` in a browser.

**If interrupted:** Just re-run the same command. Training auto-resumes from the latest snapshot.

---

## Step 7: Assemble Final Character Model

After BOTH stages complete successfully, run these commands:

```bash
mkdir data\distill_examples\my_char\character_model

copy data\distill_examples\my_char\face_morpher\checkpoint\0010\module_module.pt data\distill_examples\my_char\character_model\face_morpher.pt

copy data\distill_examples\my_char\body_morpher\checkpoint\0015\module_module.pt data\distill_examples\my_char\character_model\body_morpher.pt

copy data\images\my_char.png data\distill_examples\my_char\character_model\character.png
```

Create the model index file:

```bash
venv\Scripts\python.exe -c "
import sys; sys.path.insert(0,'src')
from tha4.charmodel.character_model import CharacterModel
CharacterModel(
    'data/distill_examples/my_char/character_model/character.png',
    'data/distill_examples/my_char/character_model/face_morpher.pt',
    'data/distill_examples/my_char/character_model/body_morpher.pt'
).save('data/distill_examples/my_char/character_model/character_model.yaml')
print('Done')
"
```

Verify the output directory contains:
```
data/distill_examples/my_char/character_model/
  character_model.yaml    (<1 KB)
  character.png           (~110-250 KB)
  face_morpher.pt         (~480 KB)
  body_morpher.pt         (~1.3 MB)
```

---

## Step 8: Install ONNX Dependencies

```bash
# Core libraries
venv\Scripts\python.exe -m pip install onnx onnxruntime simplejpeg

# GPU runtime — Windows (no cuDNN needed):
venv\Scripts\python.exe -m pip install onnxruntime-directml==1.17.1

# GPU runtime — Linux/Windows with CUDA+cuDNN:
# venv\Scripts\python.exe -m pip install onnxruntime-gpu
```

---

## Step 9: Export to model.onnx

This is the **only** export script you need. It:
1. Exports `merged_fast.onnx` (GPU-post-processed, green screen)
2. **Bakes the texture into the model** → single-input `merged_baked.onnx`
3. Copies the final artifact to `output/model.onnx`

```bash
python merge_onnx_fast.py data/distill_examples/my_char/character_model
```

**Output:**

| File | Location | Purpose |
|------|----------|---------|
| `merged_fast.onnx` | `onnx/` | Intermediate (2-input) |
| `merged_baked.onnx` | `onnx/` | Intermediate (baked) |
| **`model.onnx`** | **`output/`** | ← **Final artifact** |

**model.onnx specification:**

| Port | Name | Shape | Type | Description |
|------|------|-------|------|-------------|
| Input | `pose` | (1, 45) | float32 | 45 pose parameters |
| Output | `rgb` | (1, 3, 512, 512) | **uint8** | sRGB RGB on green (#00FF00) background |

> **Single input!** The character texture is baked into the ONNX graph as a constant. No separate image file or runtime preprocessing needed.

---

## Step 10: Test the Model

### Quick benchmark

```bash
venv\Scripts\python.exe -c "
import onnxruntime, numpy as np, time, simplejpeg
sess = onnxruntime.InferenceSession(
    'data/distill_examples/my_char/character_model/onnx/merged_fast.onnx',
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
img = np.random.randn(1,4,512,512).astype(np.float32)
pose = np.zeros((1,45), dtype=np.float32)
for _ in range(20): sess.run(None, {'image':img, 'pose':pose})
t0=time.perf_counter()
for _ in range(60): _=sess.run(None, {'image':img, 'pose':pose})[0]
t1=time.perf_counter()
print(f'FPS: {60/(t1-t0):.0f}')
"
```

Expected: 80+ fps (DirectML GPU), 3-5 fps (CPU).

### Web interactive test

```bash
python web_demo/server.py
```

Open `http://localhost:8000` — the character should follow your mouse, blink, and breathe.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `KeyError: 'WORLD_SIZE'` | Use `torch.distributed.run` wrapper, not direct `python` |
| `KeyError: 'RANK'` | Same as above — must use `torch.distributed.run` |
| `CUDA out of memory` | Reduce `batch_size` in config (face: 5, body: 8 default) |
| ONNX GPU not working | Install `onnxruntime-directml==1.17.1` (Windows, no cuDNN needed) |
| DLL load failed with onnxruntime | Ensure numpy < 2 (`pip install "numpy<2"`) |
| Training stopped unexpectedly | Re-run the same command, training auto-resumes from snapshot |
| Face looks blue/inverted | Make sure image preprocessing uses RGBA order (NOT BGRA) and premultiplied alpha |

---

## Pose Parameters Quick Reference

| Index | Name | Range | Use |
|-------|------|-------|-----|
| 18-19 | `eye_relaxed` | 0~1 | 1 = eyes fully closed (blink) |
| 26 | `mouth_aaa` | 0~1 | 0 = mouth closed |
| 37 | `iris_rotation_x` | -1~1 | Vertical eye gaze |
| 38 | `iris_rotation_y` | -1~1 | Horizontal eye gaze |
| 39 | `head_x` | -1~1 | Head tilt up/down |
| 40 | `head_y` | -1~1 | Head turn left/right |
| 42 | `body_y` | -1~1 | Body lean left/right |
| 44 | `breathing` | 0~1 | Breathing amplitude |

Full parameter list: `src/tha4/poser/modes/pose_parameters.py`

---

## Production Inference Code

```python
import numpy as np, onnxruntime as ort, simplejpeg

# 1. Load the baked model — no texture preprocessing needed!
sess = ort.InferenceSession("model.onnx",
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

# 2. Infer every frame — just pass pose
pose = np.zeros((1, 45), dtype=np.float32)
pose[0, 18] = pose[0, 19] = 1.0   # close eyes

rgb = sess.run(None, {"pose": pose})[0]  # (1, 3, 512, 512) uint8
jpeg = simplejpeg.encode_jpeg(rgb[0].transpose(1, 2, 0), quality=75)
```

**Frontend chroma-key (JS/Canvas):**
```js
const d = ctx.getImageData(0, 0, w, h).data;
for (let i = 0; i < d.length; i += 4) {
    if (d[i] < 80 && d[i+1] > 180 && d[i+2] < 80) d[i+3] = 0; // green → transparent
}
```

---

## File Inventory (this repository)

| File | Purpose |
|------|---------|
| `README.md` | English human-readable guide |
| `README_ZH.md` | Chinese human-readable guide |
| `README_AGENT.md` | This file — optimized for AI agents |
| **`merge_onnx_fast.py`** | **Export → bake → ZIP — the only script you need** |
| `bake_texture.py` | Embed character texture into ONNX graph |
| `web_demo/server.py` | Web demo server (mouse tracking + idle animation) |
| `web_demo/static/index.html` | Web demo frontend |
| `onnx_test/compare.py` | ONNX vs PyTorch validation script |

<details>
<summary>Advanced: other export scripts (not needed for normal use)</summary>

| File | Purpose |
|------|---------|
| `export_onnx.py` | Export face/body to separate ONNX files |
| `merge_onnx.py` | Merge into raw `merged.onnx` (needs CPU post-process) |

</details>

## License

- Code: MIT License
- THA4 models and images: CC BY-NC 4.0
