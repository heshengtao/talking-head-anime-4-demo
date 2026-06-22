# THA4 学生模型训练 & ONNX 导出流水线

[English](README.md) | [Agent 指引](README_AGENT.md)

> **正在使用 AI 编程助手（opencode、Claude Code 等）？**  
> 把下面这个链接发给它，即可在 AI 引导下完成全部流程：
>
> ```
> curl -fsSL https://raw.githubusercontent.com/heshengtao/talking-head-anime-4-demo/main/README_AGENT.md
> ```
>
> **Super Agent Party 用户：** 导入 `output/model.onnx`——纹理内嵌，仅需 `pose` 输入。

---

本仓库用于从单张动漫角色图片**训练轻量学生模型**，并**导出为可部署的 ZIP 包**，实现纯 ONNX GPU 实时推理——完全无需 PyTorch。

> **最终产物：** `output/model.onnx` — 纹理内嵌，单输入 `pose`，80+ fps，绿幕 `#00FF00`。前端通过色键抠绿恢复透明通道。

原始研究来自 ["Talking Head(?) Anime from a Single Image 4"](https://github.com/pkhungurn/talking-head-anime-4-demo)。本 Fork 增加了可直接用于生产环境的 ONNX 导出和 Web 演示。

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

# (Mac 专属) 转为 CoreML，M1/M2/M3/M4 加速（~74 fps vs ONNX CPU ~4 fps）
python convert_coreml.py data/distill_examples/my_char/character_model
python web_demo_mac/server.py
```

---

### 完整步骤指南

#### 第一步：搭建 Python 环境

**要求：** Python 3.10.x、Poetry 1.7+、NVIDIA GPU（RTX 2080 以上）。

```bash
# 安装 Poetry（如未安装）
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# 创建虚拟环境
git clone https://github.com/heshengtao/talking-head-anime-4-demo.git
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
2. 新建图层，填充**黑色**
3. 用**白色矩形**涂抹**眼睛和嘴巴**区域
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
运行导出（推荐方案——GPU 后处理内嵌，自动打包）：

```bash
python merge_onnx_fast.py data/distill_examples/my_char/character_model
```

输出：
- `data/distill_examples/my_char/character_model/onnx/merged_fast.onnx`（约 4.5 MB）——绿幕抠图模型
- **`output/model.onnx`**——最终产物（单输入，纹理内嵌）

**模型输入输出（merged_fast.onnx）：**

| 端口 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `image` | (1, 4, 512, 512) | float32 | 预处理后的纹理图，[-1,1]，预乘 alpha |
| 输入 | `pose` | (1, 45) | float32 | 45 维姿态参数 |
| 输出 | `rgb` | (1, 3, 512, 512) | **uint8** | sRGB 图像，绿色背景（#00FF00） |

> 绿幕 `#00FF00`——前端通过色键抠绿恢复透明。深色衣服不受影响。

#### 第九步：（Mac 专属）转为 Apple Silicon CoreML

在 **M1/M2/M3/M4 Mac** 上部署时，可将 PyTorch 模型直接转为 CoreML `.mlpackage`——无需 ONNX、无需 GPU，原生 Neural Engine 推理。

**性能：** M4 约 74 fps（比 ONNX CPU 快 18 倍）。

```bash
# 1. 安装 CoreML 工具
pip install coremltools pillow

# 2. 转换（单输入模型，纹理内嵌）
python convert_coreml.py data/distill_examples/my_char/character_model

# 输出：output/model.mlpackage
```

**model.mlpackage 规范：**

| 端口 | 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|------|
| 输入 | `pose` | (1, 45) | float32 | 45 维姿态参数 |
| 输出 | `blended` | (1, 4, 512, 512) | float32 | RGBA 混合图像，[-1,1]，预乘 alpha |

**推理代码（无 PyTorch 依赖）：**

```python
from coremltools.models import MLModel
model = MLModel("model.mlpackage")
output = model.predict({"pose": pose_array})
```

**Mac Web 演示：**

```bash
python web_demo_mac/server.py
# 打开 http://localhost:8000 — 角色跟随鼠标
```

该 demo 在 Mac 上自动选择 CoreML 后端，若无 `.mlpackage` 则回退 ONNX CPU。

#### 第十步：测试 ONNX 模型

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

### 生产环境使用

```python
import numpy as np, onnxruntime as ort, simplejpeg

sess = ort.InferenceSession("model.onnx",
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

pose = np.zeros((1, 45), dtype=np.float32)
pose[0,18] = pose[0,19] = 1.0   # 闭眼

rgb = sess.run(None, {"pose": pose})[0]  # (1,3,512,512) uint8
jpeg = simplejpeg.encode_jpeg(rgb[0].transpose(1,2,0), quality=75)
```

---

### 许可证

- 代码：MIT License
- `data/images/` 下的 THA4 模型和图片：CC BY-NC 4.0
