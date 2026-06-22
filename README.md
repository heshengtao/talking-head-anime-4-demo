# THA4 Student Model Training & ONNX Export Pipeline

[中文版](README_ZH.md) | [Agent Instructions](README_AGENT.md)

> **Using an AI coding agent (opencode, Claude Code, etc.)?**  
> Send it the raw URL of the agent instructions:
>
> ```
> curl -fsSL https://raw.githubusercontent.com/heshengtao/talking-head-anime-4-demo/main/README_AGENT.md
> ```
>
> **Super Agent Party users:** Import `output/model.onnx` — texture baked in, single `pose` input.

---

This repository contains tools to **train a lightweight student model** from a single anime character image, then **export it to a deployable ZIP** for real-time GPU inference — no PyTorch dependency at runtime.

> **Final output:** `output/model.onnx` — texture baked in, single `pose` input, 80+ fps, green-screen `#00FF00`. Frontends chroma-key the green to restore transparency.

The original research is from ["Talking Head(?) Anime from a Single Image 4"](https://github.com/pkhungurn/talking-head-anime-4-demo). This fork adds production-ready ONNX export and a web demo.

### Quickstart for Agents / Developers

```bash
# 1. Clone
git clone https://github.com/heshengtao/talking-head-anime-4-demo.git
cd talking-head-anime-4-demo

# 2. Install Python environment
python -m venv venv --prompt talking-head-anime-4-demo
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
cd poetry && poetry install && cd ..

# 3. Download base models (required)
#    Download from: https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1
#    Extract to → data/tha4/ (contains body_morpher.pt, face_morpher.pt, etc.)

# 4. Download pose dataset (required for training)
#    Download from: https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1
#    Save to → data/pose_dataset.pt

# 5. Prepare your character image and face mask (see Constraints below)

# 6. Train your student model
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_face_morpher.py --config_file data/distill_examples/my_char/config.yaml
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_body_morpher.py --config_file data/distill_examples/my_char/config.yaml

# 7. Assemble and export to ONNX (see Full Steps below)
python merge_onnx_fast.py data/distill_examples/my_char/character_model

# 8. Test
python web_demo/server.py
# Open http://localhost:8000

# (Mac only) Convert to CoreML for M1/M2/M3/M4 (~74 fps vs ~4 fps ONNX CPU)
python convert_coreml.py data/distill_examples/my_char/character_model
python web_demo_mac/server.py
```

---

### Full Step-by-Step Guide

#### Step 1: Set Up Python Environment

**Requirements:** Python 3.10.x, Poetry 1.7+, NVIDIA GPU (RTX 2080+).

```bash
# Install Poetry (if missing)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Create venv
git clone https://github.com/heshengtao/talking-head-anime-4-demo.git
cd talking-head-anime-4-demo
python -m venv venv --prompt talking-head-anime-4-demo

# Activate
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
cd poetry && poetry install && cd ..
```

The `pyproject.toml` installs PyTorch 1.13.1+cu117, wxPython, OpenCV, MediaPipe, and all other requirements.

#### Step 2: Download Required Files

**THA4 base models** (Dropbox, ~200 MB):
```
URL:  https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1
Extract to → data/tha4/
```

Expected files:
```
data/tha4/
  body_morpher.pt
  eyebrow_decomposer.pt
  eyebrow_morphing_combiner.pt
  face_morpher.pt
  upscaler.pt
```

**Pose dataset** (Dropbox, ~200 MB):
```
URL:  https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1
Save to → data/pose_dataset.pt
```

#### Step 3: Prepare Character Image

See [Constraints on Input Images](#constraints-on-input-images) below.

Save your image as:
```
data/images/my_char.png
```

#### Step 4: Create Face Mask Image

Use [GIMP](https://www.gimp.org/) (free, open-source) to create a face mask:

1. Open your `my_char.png` in GIMP
2. Create a new layer, fill it **white**
3. Use the paintbrush (black color) to paint over the **eyes and mouth** areas
4. Export as `my_char_face_mask.png` (512×512, RGB, no alpha)

Reference: `data/images/lambda_00_face_mask.png` and `data/images/lambda_00_face_mask.xcf`.

Save to:
```
data/images/my_char_face_mask.png
```

The mask must be a black-and-white image where:
- White (255,255,255) = areas the model CAN modify
- Black (0,0,0) = eyes and mouth regions (protected)

#### Step 5: Create Training Configuration

Create `data/distill_examples/my_char/config.yaml`:

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

Create the workspace directory:
```bash
mkdir data\distill_examples\my_char
```

#### Step 6: Train the Student Model

Training consists of two stages that run sequentially:

**Stage 1 — Face Morpher** (~1,000,000 examples, ~3 hours on RTX 4080):
```bash
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_face_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

**Stage 2 — Body Morpher** (~1,500,000 examples, ~5 hours on RTX 4080):
```bash
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_body_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

> **Monitoring:** Run `tensorboard --logdir=data/distill_examples/my_char` and open `http://localhost:6006`.

> **Resuming:** If training is interrupted, just re-run the same command. The trainer auto-resumes from the latest snapshot.

> **Alternative GUI:** `bin\run.bat src\tha4\app\distiller_ui.py` provides a graphical interface for creating configs and launching training.

#### Step 7: Assemble the Final Character Model

After both stages complete, copy the final checkpoints:

```bash
# Create output directory
mkdir data\distill_examples\my_char\character_model

# Copy final face morpher (checkpoint 0010 = 1,000,000 examples)
copy data\distill_examples\my_char\face_morpher\checkpoint\0010\module_module.pt data\distill_examples\my_char\character_model\face_morpher.pt

# Copy final body morpher (checkpoint 0015 = 1,500,000 examples)
copy data\distill_examples\my_char\body_morpher\checkpoint\0015\module_module.pt data\distill_examples\my_char\character_model\body_morpher.pt

# Copy character image
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

Final structure:
```
data/distill_examples/my_char/character_model/
  character_model.yaml
  character.png        (~110-250 KB)
  face_morpher.pt      (~480 KB)
  body_morpher.pt      (~1.3 MB)
```

#### Step 8: Export to ONNX

Install ONNX dependencies:
```bash
pip install onnx onnxruntime simplejpeg
# For GPU: pip install onnxruntime-directml==1.17.1  (Windows, no cuDNN needed)
# Or:       pip install onnxruntime-gpu                (Linux/Win, requires CUDA+cuDNN)
```
Run the export (recommended — GPU post-processing baked in, auto-packaged):

```bash
python merge_onnx_fast.py data/distill_examples/my_char/character_model
```

Output:
- `data/distill_examples/my_char/character_model/onnx/merged_fast.onnx` (~4.5 MB) — green-screen model
- **`output/model.onnx`** — final artifact (single-input, texture baked in)

**Model I/O (merged_fast.onnx):**

| Port | Name | Shape | Type | Description |
|------|------|-------|------|-------------|
| Input | `image` | (1, 4, 512, 512) | float32 | Preprocessed texture in [-1,1] with premultiplied alpha |
| Input | `pose` | (1, 45) | float32 | 45 pose parameters |
| Output | `rgb` | (1, 3, 512, 512) | **uint8** | sRGB RGB on green (#00FF00) background |

> Green screen `#00FF00` — frontends chroma-key the green to restore transparency. Dark clothing is not affected.

#### Step 9: (Mac Only) Convert to Apple Silicon CoreML

For **M1/M2/M3/M4 Mac** deployment, convert the PyTorch model directly to CoreML `.mlpackage` — no ONNX, no GPU dependency, native Neural Engine acceleration.

**Performance:** ~74 fps on M4 (18× faster than ONNX CPU).

```bash
# 1. Install CoreML tooling
pip install coremltools pillow

# 2. Convert (single-input model, texture baked in)
python convert_coreml.py data/distill_examples/my_char/character_model

# Output: output/model.mlpackage
```

**model.mlpackage specification:**

| Port | Name | Shape | Type | Description |
|------|------|-------|------|-------------|
| Input | `pose` | (1, 45) | float32 | 45 pose parameters |
| Output | `blended` | (1, 4, 512, 512) | float32 | RGBA blended image in [-1, 1], premultiplied alpha |

**Inference code (no PyTorch dependency):**

```python
from coremltools.models import MLModel
model = MLModel("model.mlpackage")
output = model.predict({"pose": pose_array})
```

**Mac web demo:**

```bash
python web_demo_mac/server.py
# Open http://localhost:8000 — character follows mouse
```

This demo auto-selects CoreML on Mac, falling back to ONNX CPU if `.mlpackage` is unavailable.

#### Step 10: Test the ONNX Model

**Command-line benchmark:**
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

Expected: 80+ fps with DirectML GPU (Windows), 3-5 fps with CPU.

**Web interactive test:**
```bash
python web_demo/server.py
```
Open `http://localhost:8000` — character follows mouse, blinks, breathes.

---

### Constraints on Input Images

The input character image must follow these rules for the model to work well:

- **Resolution:** Exactly 512 × 512 pixels (programs will resize automatically but results are best at 512×512).
- **Alpha channel:** Must be RGBA with transparent background (alpha = 0 for all background pixels).
- **Character count:** Only one humanoid character.
- **Pose:** Character must stand upright facing forward.
- **Hands:** Below and far from the head.
- **Head position:** The head should roughly fit in a 128 × 128 box at the middle of the top half of the image.
- **Background:** All background pixels must have alpha = 0.

![Example of a valid input image](docs/images/input_spec.png)

---

### Pose Parameters Reference

The 45-dimensional pose vector controls the character's expression and pose:

| Index | Name | Description | Range |
|-------|------|-------------|-------|
| 0-1 | `eyebrow_troubled` | Troubled eyebrows (L/R) | 0~1 |
| 2-3 | `eyebrow_angry` | Angry eyebrows (L/R) | 0~1 |
| 4-5 | `eyebrow_lowered` | Lowered eyebrows (L/R) | 0~1 |
| 6-7 | `eyebrow_raised` | Raised eyebrows (L/R) | 0~1 |
| 8-9 | `eyebrow_happy` | Happy eyebrows (L/R) | 0~1 |
| 10-11 | `eyebrow_serious` | Serious eyebrows (L/R) | 0~1 |
| 12-13 | `eye_wink` | Eye wink (L/R) | 0~1 |
| 14-15 | `eye_happy_wink` | Happy wink (L/R) | 0~1 |
| 16-17 | `eye_surprised` | Surprised eyes (L/R) | 0~1 |
| **18-19** | **`eye_relaxed`** | **Close eyes (L/R), 1=fully closed** | 0~1 |
| 20-21 | `eye_unimpressed` | Unimpressed eyes (L/R) | 0~1 |
| 22-23 | `eye_raised_lower_eyelid` | Raised lower eyelid (L/R) | 0~1 |
| 24-25 | `iris_small` | Small iris (L/R) | 0~1 |
| **26** | **`mouth_aaa`** | **Open mouth, 0=closed** | 0~1 |
| 27 | `mouth_iii` | Wide mouth (vowel "i") | 0~1 |
| 28 | `mouth_uuu` | Pursed mouth (vowel "u") | 0~1 |
| 29 | `mouth_eee` | Wide mouth (vowel "e") | 0~1 |
| 30 | `mouth_ooo` | Round mouth (vowel "o") | 0~1 |
| 31 | `mouth_delta` | Mouth delta | 0~1 |
| 32-33 | `mouth_lowered_corner` | Lowered mouth corners (L/R) | 0~1 |
| 34-35 | `mouth_raised_corner` | Raised mouth corners (L/R) | 0~1 |
| 36 | `mouth_smirk` | Smirk | 0~1 |
| **37** | **`iris_rotation_x`** | **Eye vertical gaze** | -1~1 |
| **38** | **`iris_rotation_y`** | **Eye horizontal gaze** | -1~1 |
| **39** | **`head_x`** | **Head tilt up/down** | -1~1 |
| **40** | **`head_y`** | **Head turn left/right** | -1~1 |
| 41 | `neck_z` | Neck extension | -1~1 |
| **42** | **`body_y`** | **Body lean left/right** | -1~1 |
| 43 | `body_z` | Body lean forward/back | -1~1 |
| **44** | **`breathing`** | **Breathing amplitude** | 0~1 |

Source: `src/tha4/poser/modes/pose_parameters.py`.

---

### Using the Deployable ONNX in Production

```python
import numpy as np, onnxruntime as ort, simplejpeg

sess = ort.InferenceSession("model.onnx",
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

pose = np.zeros((1, 45), dtype=np.float32)
pose[0,18] = pose[0,19] = 1.0   # close eyes

rgb = sess.run(None, {"pose": pose})[0]  # (1,3,512,512) uint8
jpeg = simplejpeg.encode_jpeg(rgb[0].transpose(1,2,0), quality=75)
```

---

### License

- Code: MIT License
- THA4 models and images under `data/images/`: CC BY-NC 4.0
