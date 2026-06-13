# THA4 Student Model Training & ONNX Export Pipeline

[English](#english) | [中文](#中文)

---

<a id="english"></a>
## English

This repository contains tools to **train a lightweight student model** from a single anime character image, then **export it to ONNX** for real-time GPU inference — no PyTorch dependency at runtime.

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
```

---

### Full Step-by-Step Guide

#### Step 1: Set Up Python Environment

**Requirements:** Python 3.10.x, Poetry 1.7+, NVIDIA GPU (RTX 2080+).

```bash
# Install Poetry (if missing)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Create venv
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

Run the export (recommended — GPU post-processing baked in):
```bash
python merge_onnx_fast.py data/distill_examples/my_char/character_model
```

Output: `data/distill_examples/my_char/character_model/onnx/merged_fast.onnx` (~4.5 MB).

**Model I/O:**

| Port | Name | Shape | Type | Description |
|------|------|-------|------|-------------|
| Input | `image` | (1, 4, 512, 512) | float32 | Preprocessed character image in [-1,1] with premultiplied alpha |
| Input | `pose` | (1, 45) | float32 | 45 pose parameters (see table below) |
| Output | `rgb` | (1, 3, 512, 512) | **uint8** | Final RGB image, sRGB, composited on dark background |

> No CPU post-processing needed — the ONNX graph handles un-premultiply, sRGB conversion, and background compositing on GPU.

**Alternative exports:**
| Script | Output | Notes |
|--------|--------|-------|
| `export_onnx.py <dir>` | `face_morpher.onnx` + `body_morpher.onnx` | Two separate models |
| `merge_onnx.py <dir>` | `merged.onnx` | Single model, raw [-1,1] output, needs CPU post-processing |
| `merge_onnx_fast.py <dir>` | `merged_fast.onnx` | **Recommended** — uint8 RGB output, GPU post-processing baked in |

#### Step 9: Test the ONNX Model

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

<a id="constraints-on-input-images"></a>
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

### Using merged_fast.onnx in Production

```python
import numpy as np
import onnxruntime as ort
import simplejpeg
from PIL import Image

# 1. Load model (DirectML for Windows GPU, CUDA for Linux)
sess = ort.InferenceSession("merged_fast.onnx",
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

# 2. Preprocess character image (once at startup)
def load_image(path):
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    # sRGB → linear
    rgb = img[:,:,:3].copy()
    m = rgb <= 0.04045
    rgb[m] /= 12.92
    rgb[~m] = ((rgb[~m] + 0.055) / 1.055) ** 2.4
    img[:,:,:3] = rgb
    # premultiply alpha
    img[:,:,:3] *= img[:,:,3:4]
    # [0,1] → [-1,1]
    img = img * 2.0 - 1.0
    return img.transpose(2,0,1)[None].astype(np.float32)

image_np = load_image("character.png")

# 3. Infer every frame
pose = np.zeros((1, 45), dtype=np.float32)
pose[0,18] = pose[0,19] = 1.0   # close eyes
pose[0,44] = 0.3                 # breathing

rgb = sess.run(None, {"image": image_np, "pose": pose})[0]  # (1,3,512,512) uint8
jpeg = simplejpeg.encode_jpeg(rgb[0].transpose(1,2,0), quality=75)
```

---

<a id="中文"></a>
## 中文

本仓库用于从单张动漫角色图片**训练轻量学生模型**，并**导出为 ONNX** 实现实时 GPU 推理——运行时无需 PyTorch。

原始研究来自["Talking Head(?) Anime from a Single Image 4"](https://github.com/pkhungurn/talking-head-anime-4-demo)。本 Fork 增加了可直接用于生产环境的 ONNX 导出和 Web 演示。

### 快速开始（Agent / 开发者用）

```bash
# 1. 克隆仓库
git clone https://github.com/heshengtao/talking-head-anime-4-demo.git
cd talking-head-anime-4-demo

# 2. 安装 Python 环境
python -m venv venv --prompt talking-head-anime-4-demo
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
cd poetry && poetry install && cd ..

# 3. 下载基础模型（必须）
#    从 https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1 下载
#    解压到 → data/tha4/（包含 body_morpher.pt、face_morpher.pt 等）

# 4. 下载姿态数据集（训练用，必须）
#    从 https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1 下载
#    保存到 → data/pose_dataset.pt

# 5. 准备角色图片和面部遮罩（见下文约束条件）

# 6. 训练学生模型
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_face_morpher.py --config_file data/distill_examples/my_char/config.yaml
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_body_morpher.py --config_file data/distill_examples/my_char/config.yaml

# 7. 组装并导出 ONNX
python merge_onnx_fast.py data/distill_examples/my_char/character_model

# 8. 测试
python web_demo/server.py
# 打开 http://localhost:8000
```

---

### 完整步骤指南

#### 第一步：搭建 Python 环境

**要求:** Python 3.10.x、Poetry 1.7+、NVIDIA GPU（RTX 2080 以上）。

```bash
# 安装 Poetry（如未安装）
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# 创建虚拟环境
cd talking-head-anime-4-demo
python -m venv venv --prompt talking-head-anime-4-demo

# 激活
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 安装依赖
cd poetry && poetry install && cd ..
```

`pyproject.toml` 会自动安装 PyTorch 1.13.1+cu117、wxPython、OpenCV、MediaPipe 等全部依赖。

#### 第二步：下载必要文件

**THA4 基础模型**（Dropbox，约 200 MB）：
```
下载地址：https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1
解压到 → data/tha4/
```

预期文件：
```
data/tha4/
  body_morpher.pt
  eyebrow_decomposer.pt
  eyebrow_morphing_combiner.pt
  face_morpher.pt
  upscaler.pt
```

**姿态数据集**（Dropbox，约 200 MB）：
```
下载地址：https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1
保存到 → data/pose_dataset.pt
```

#### 第三步：准备角色图片

请参见下文 [输入图片约束条件](#输入图片约束条件)。

将图片保存为：
```
data/images/my_char.png
```

#### 第四步：创建面部遮罩图片

使用 [GIMP](https://www.gimp.org/)（免费开源软件）创建面部遮罩：

1. 用 GIMP 打开 `my_char.png`
2. 新建图层，填充**白色**
3. 用画笔工具（黑色）涂抹**眼睛和嘴巴**区域
4. 导出为 `my_char_face_mask.png`（512×512，RGB，无透明通道）

参考文件：`data/images/lambda_00_face_mask.png` 和 `data/images/lambda_00_face_mask.xcf`。

保存到：
```
data/images/my_char_face_mask.png
```

遮罩必须是黑白图像：
- 白色 (255,255,255) = 模型可以修改的区域
- 黑色 (0,0,0) = 眼睛和嘴巴区域（受保护）

#### 第五步：创建训练配置

创建 `data/distill_examples/my_char/config.yaml`：

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

创建工作目录：
```bash
mkdir data\distill_examples\my_char
```

#### 第六步：训练学生模型

训练分两个阶段，顺序执行：

**阶段一 — 面部模型**（约 100 万样本，RTX 4080 约 3 小时）：
```bash
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_face_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

**阶段二 — 身体模型**（约 150 万样本，RTX 4080 约 5 小时）：
```bash
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_body_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

> **监控进度：** 运行 `tensorboard --logdir=data/distill_examples/my_char`，打开 `http://localhost:6006`。

> **断点续训：** 如果训练中断，直接重新运行相同命令，训练框架自动从最新 snapshot 恢复。

> **可选 GUI：** `bin\run.bat src\tha4\app\distiller_ui.py` 提供图形化配置和训练界面。

#### 第七步：组装最终角色模型

两个阶段都完成后，复制最终 checkpoint：

```bash
# 创建输出目录
mkdir data\distill_examples\my_char\character_model

# 复制最终面部模型（checkpoint 0010 = 100 万样本）
copy data\distill_examples\my_char\face_morpher\checkpoint\0010\module_module.pt data\distill_examples\my_char\character_model\face_morpher.pt

# 复制最终身体模型（checkpoint 0015 = 150 万样本）
copy data\distill_examples\my_char\body_morpher\checkpoint\0015\module_module.pt data\distill_examples\my_char\character_model\body_morpher.pt

# 复制角色原图
copy data\images\my_char.png data\distill_examples\my_char\character_model\character.png
```

创建模型索引文件：
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

最终结构：
```
data/distill_examples/my_char/character_model/
  character_model.yaml
  character.png        (~110-250 KB)
  face_morpher.pt      (~480 KB)
  body_morpher.pt      (~1.3 MB)
```

#### 第八步：导出 ONNX

安装 ONNX 依赖：
```bash
pip install onnx onnxruntime simplejpeg
# GPU 推理（Windows，无需 cuDNN）：
pip install onnxruntime-directml==1.17.1
# GPU 推理（Linux/Windows，需要 CUDA+cuDNN）：
pip install onnxruntime-gpu
```

运行导出（推荐方案——GPU 后处理内嵌）：
```bash
python merge_onnx_fast.py data/distill_examples/my_char/character_model
```

输出：`data/distill_examples/my_char/character_model/onnx/merged_fast.onnx`（约 4.5 MB）。

**模型输入输出：**

| 端口 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `image` | (1, 4, 512, 512) | float32 | 预处理后的角色图，[-1,1]，预乘 alpha |
| 输入 | `pose` | (1, 45) | float32 | 45 维姿态参数 |
| 输出 | `rgb` | (1, 3, 512, 512) | **uint8** | 最终 RGB 图像，sRGB，已复合深色背景 |

> 无需 CPU 后处理——ONNX 图内已包含反除 alpha、sRGB 转换和背景复合，全部在 GPU 上完成。

**其他导出方案：**
| 脚本 | 输出 | 说明 |
|------|------|------|
| `export_onnx.py <dir>` | `face_morpher.onnx` + `body_morpher.onnx` | 两个独立模型 |
| `merge_onnx.py <dir>` | `merged.onnx` | 单一模型，原始 [-1,1] 输出，需 CPU 后处理 |
| `merge_onnx_fast.py <dir>` | `merged_fast.onnx` | **推荐** — uint8 RGB 输出，GPU 后处理内嵌 |

#### 第九步：测试 ONNX 模型

**命令行性能测试：**
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

预期：DirectML GPU 80+ fps（Windows），CPU 3-5 fps。

**Web 交互测试：**
```bash
python web_demo/server.py
```
打开 `http://localhost:8000`——角色跟随鼠标、眨眼、呼吸。

---

<a id="输入图片约束条件"></a>
### 输入图片约束条件

为了让系统正常工作，输入图片必须满足以下约束：

- **分辨率：** 必须为 512 × 512 像素（程序会自动缩放，但 512×512 效果最佳）。
- **透明通道：** 必须是 RGBA 格式，背景透明（所有背景像素的 alpha = 0）。
- **角色数量：** 只能包含一个类人角色。
- **姿势：** 角色必须直立朝前。
- **手部：** 手部应在头部下方且远离头部。
- **头部位置：** 头部应大致位于图像上半部中央的 128 × 128 区域内。
- **背景：** 所有背景像素的 alpha 值必须为 0。

![合规输入图片示例](docs/images/input_spec.png)

---

### 姿态参数速查

45 维姿态向量，控制角色表情和姿态：

| 索引 | 名称 | 说明 | 范围 |
|------|------|------|------|
| 0-1 | `eyebrow_troubled` | 困扰眉（左/右） | 0~1 |
| 2-3 | `eyebrow_angry` | 愤怒眉（左/右） | 0~1 |
| 4-5 | `eyebrow_lowered` | 下压眉（左/右） | 0~1 |
| 6-7 | `eyebrow_raised` | 上扬眉（左/右） | 0~1 |
| 8-9 | `eyebrow_happy` | 开心眉（左/右） | 0~1 |
| 10-11 | `eyebrow_serious` | 严肃眉（左/右） | 0~1 |
| 12-13 | `eye_wink` | 眨眼（左/右） | 0~1 |
| 14-15 | `eye_happy_wink` | 笑眼（左/右） | 0~1 |
| 16-17 | `eye_surprised` | 惊讶眼（左/右） | 0~1 |
| **18-19** | **`eye_relaxed`** | **闭眼（左/右），1=全闭** | 0~1 |
| 20-21 | `eye_unimpressed` | 冷漠眼（左/右） | 0~1 |
| 22-23 | `eye_raised_lower_eyelid` | 下眼皮上提（左/右） | 0~1 |
| 24-25 | `iris_small` | 瞳孔缩小（左/右） | 0~1 |
| **26** | **`mouth_aaa`** | **张嘴，0=闭嘴** | 0~1 |
| 27 | `mouth_iii` | 咧嘴（元音 i） | 0~1 |
| 28 | `mouth_uuu` | 噘嘴（元音 u） | 0~1 |
| 29 | `mouth_eee` | 咧宽嘴（元音 e） | 0~1 |
| 30 | `mouth_ooo` | 圆嘴（元音 o） | 0~1 |
| 31 | `mouth_delta` | 嘴变化量 | 0~1 |
| 32-33 | `mouth_lowered_corner` | 下压嘴角（左/右） | 0~1 |
| 34-35 | `mouth_raised_corner` | 上扬嘴角（左/右） | 0~1 |
| 36 | `mouth_smirk` | 撇嘴 | 0~1 |
| **37** | **`iris_rotation_x`** | **眼球垂直注视** | -1~1 |
| **38** | **`iris_rotation_y`** | **眼球水平注视** | -1~1 |
| **39** | **`head_x`** | **头部上下点头** | -1~1 |
| **40** | **`head_y`** | **头部左右转动** | -1~1 |
| 41 | `neck_z` | 颈部伸缩 | -1~1 |
| **42** | **`body_y`** | **身体左右倾** | -1~1 |
| 43 | `body_z` | 身体前后倾 | -1~1 |
| **44** | **`breathing`** | **呼吸幅度** | 0~1 |

来源：`src/tha4/poser/modes/pose_parameters.py`。

---

### 生产环境使用 merged_fast.onnx

```python
import numpy as np
import onnxruntime as ort
import simplejpeg
from PIL import Image

# 1. 加载模型（Windows 用 DirectML GPU，Linux 用 CUDA）
sess = ort.InferenceSession("merged_fast.onnx",
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

# 2. 预处理角色图片（启动时执行一次即可）
def load_image(path):
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    # sRGB → linear
    rgb = img[:,:,:3].copy()
    m = rgb <= 0.04045
    rgb[m] /= 12.92
    rgb[~m] = ((rgb[~m] + 0.055) / 1.055) ** 2.4
    img[:,:,:3] = rgb
    # premultiply alpha
    img[:,:,:3] *= img[:,:,3:4]
    # [0,1] → [-1,1]
    img = img * 2.0 - 1.0
    return img.transpose(2,0,1)[None].astype(np.float32)

image_np = load_image("character.png")

# 3. 每帧推理
pose = np.zeros((1, 45), dtype=np.float32)
pose[0,18] = pose[0,19] = 1.0   # 闭眼
pose[0,44] = 0.3                 # 呼吸

rgb = sess.run(None, {"image": image_np, "pose": pose})[0]  # (1,3,512,512) uint8
jpeg = simplejpeg.encode_jpeg(rgb[0].transpose(1,2,0), quality=75)
```

---

### License

- Code: MIT License
- THA4 models and images under `data/images/`: CC BY-NC 4.0

[Back to top](#tha4-student-model-training--onnx-export-pipeline)
