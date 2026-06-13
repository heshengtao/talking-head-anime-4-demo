# THA4 蒸馏模型训练 & 导出 ONNX 完整指南

从一张角色图片到一个可在 Web 端实时渲染的 `merged_fast.onnx` 模型。

---

## 前置条件

- **GPU**: NVIDIA RTX 2080 或更新（推荐 RTX 4080+）
- **OS**: Windows 10/11（Linux 也可，本文以 Windows 为例）
- **磁盘**: ~30 GB（含依赖、模型、训练中间文件）
- **Python**: 3.10.x
- **时间**: 训练约 30 小时（RTX A6000）/ ~8-10 小时（RTX 4080）

---

## 第一步：克隆仓库并安装依赖

```powershell
git clone https://github.com/pkhungurn/talking-head-anime-4-demo.git
cd talking-head-anime-4-demo
```

```powershell
# 安装 Poetry（如果没有）
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# 创建虚拟环境
python -m venv venv --prompt talking-head-anime-4-demo
venv\Scripts\activate
cd poetry
poetry install
cd ..
```

---

## 第二步：下载模型和数据

### THA4 基础模型（必须）

从 [Dropbox](https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1) 下载 `tha4-models.zip`，解压到 `data/tha4/`：

```
data/tha4/
  body_morpher.pt
  eyebrow_decomposer.pt
  eyebrow_morphing_combiner.pt
  face_morpher.pt
  upscaler.pt
```

### 姿态数据集（训练用，必须）

从 [Dropbox](https://www.dropbox.com/scl/fi/du10e6buzr5bslbe025qu/pose_dataset.pt?rlkey=y052g4n3xb14nu2elctzouc5x&dl=1) 下载 `pose_dataset.pt`，放到 `data/`：

```
data/pose_dataset.pt
```

---

## 第三步：准备角色图片和面部遮罩

### 角色图片要求

- **512×512 RGBA PNG**
- 角色直立朝前，双手在头部以下
- 头部大致在图像上半部中央 128×128 区域内
- 背景透明（alpha=0）

### 面部遮罩图片

- **512×512 黑白 PNG**（RGB 三通道，每像素只能是 0 或 255）
- 用黑色覆盖眼睛和嘴巴区域
- 可以用 [GIMP](https://www.gimp.org/) 制作，参考 `data/images/lambda_00_face_mask.xcf`

将图片放到 `data/images/`：
```
data/images/my_char.png
data/images/my_char_face_mask.png
```

---

## 第四步：训练学生模型

### 方式一：图形界面（推荐新手）

```powershell
bin\run.bat src\tha4\app\distiller_ui.py
```

在界面中：
1. 设置 **Character Image** → `data/images/my_char.png`
2. 设置 **Face Mask** → `data/images/my_char_face_mask.png`
3. 设置 **Prefix (workspace)** → `data/distill_examples/my_char`
4. 点击 **Run** 开始训练

训练分两个阶段：
- **face_morpher**：约 100 万样本（先跑）
- **body_morpher**：约 150 万样本（后跑）

### 方式二：命令行

先手动创建配置文件 `data/distill_examples/my_char/config.yaml`：

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

然后依次运行：

```powershell
# 训练面部模型
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_face_morpher.py --config_file data/distill_examples/my_char/config.yaml

# 训练身体模型
venv\Scripts\python.exe -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --standalone src\tha4\distiller\distill_body_morpher.py --config_file data/distill_examples/my_char/config.yaml
```

### 监控训练进度

```powershell
venv\Scripts\python.exe -m tensorboard.main --logdir=data/distill_examples/my_char
```

浏览器打开 `http://localhost:6006`。

### 断点续训

如果训练中断，直接重新运行上面的命令即可，训练框架自动从最新 snapshot 恢复。

---

## 第五步：组装最终模型

训练完成后，手动拼接最终学生模型：

```powershell
# 复制最终的 checkpoint 文件
copy data\distill_examples\my_char\face_morpher\checkpoint\0010\module_module.pt data\distill_examples\my_char\character_model\face_morpher.pt
copy data\distill_examples\my_char\body_morpher\checkpoint\0015\module_module.pt data\distill_examples\my_char\character_model\body_morpher.pt
copy data\images\my_char.png data\distill_examples\my_char\character_model\character.png
```

创建索引文件：

```powershell
venv\Scripts\python.exe -c "
import sys; sys.path.insert(0,'src')
from tha4.charmodel.character_model import CharacterModel
cm = CharacterModel(
    'data/distill_examples/my_char/character_model/character.png',
    'data/distill_examples/my_char/character_model/face_morpher.pt',
    'data/distill_examples/my_char/character_model/body_morpher.pt')
cm.save('data/distill_examples/my_char/character_model/character_model.yaml')
print('Done')
"
```

---

## 第六步：导出 ONNX 模型

### 6.1 安装 ONNX 依赖

```powershell
venv\Scripts\python.exe -m pip install onnx onnxruntime onnxruntime-gpu simplejpeg
```

> `onnxruntime-gpu` 需要 cuDNN。如果你的环境没有 cuDNN，可以用 `onnxruntime-directml`（仅 Windows）：
> ```
> pip install onnxruntime-directml==1.17.1 simplejpeg
> ```

### 6.2 导出合并模型（GPU 后处理版本）

```powershell
python merge_onnx_fast.py data/distill_examples/my_char/character_model
```

输出：`data/distill_examples/my_char/character_model/onnx/merged_fast.onnx`

这个模型的特点：
- **输入**：`image` (1,4,512,512) float32, `pose` (1,45) float32
- **输出**：`rgb` (1,3,512,512) **uint8** — 已经是 sRGB、已复合深色背景
- **零 CPU 后处理**：反除 alpha、sRGB 转换、背景复合全在 GPU 内完成
- **文件大小**：约 4.5 MB

### 6.3 （可选）导出分体 ONNX 模型

```powershell
python export_onnx.py data/distill_examples/my_char/character_model
```

输出：
- `face_morpher.onnx` — 面部变形（~0.6 MB）
- `body_morpher.onnx` — 身体变形（~3.9 MB）

### 6.4 （可选）导出无后处理的合并模型

```powershell
python merge_onnx.py data/distill_examples/my_char/character_model
```

输出：`merged.onnx` — 合并但保留原始 [-1,1] 输出（需 CPU 后处理）

---

## 第七步：测试 ONNX 模型

### 7.1 快速命令行测试

```powershell
venv\Scripts\python.exe -c "
import onnxruntime, numpy as np, time, simplejpeg

sess = onnxruntime.InferenceSession(
    'data/distill_examples/my_char/character_model/onnx/merged_fast.onnx',
    providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

img = np.random.randn(1,4,512,512).astype(np.float32)
pose = np.zeros((1,45), dtype=np.float32)

# warmup
for _ in range(10): sess.run(None, {'image':img, 'pose':pose})

# benchmark
t0 = time.perf_counter()
for _ in range(60):
    out = sess.run(None, {'image':img, 'pose':pose})[0]
    jpg = simplejpeg.encode_jpeg(out[0].transpose(1,2,0), quality=75)
t1 = time.perf_counter()

print(f'FPS: {60/(t1-t0):.0f}')
print(f'Output shape: {out.shape}, dtype: {out.dtype}')
"
```

预期：80+ fps（DirectML GPU）或 3-5 fps（CPU）。

### 7.2 Web 交互测试

启动测试服务器：

```powershell
python web_demo/server.py
```

浏览器打开 `http://localhost:8000`，移动鼠标看角色跟随效果。

服务器会自动选择最佳后端：
1. ONNX DirectML GPU（需要 `onnxruntime-directml`，Windows 自带 DirectX）
2. PyTorch CUDA GPU（fallback，需要 torch）
3. CPU（最后 fallback）

---

## 模型文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `face_morpher.pt` | ~480 KB | 面部变形（PyTorch） |
| `body_morpher.pt` | ~1.3 MB | 身体变形（PyTorch） |
| `character.png` | ~110-250 KB | 角色原图 |
| `character_model.yaml` | <1 KB | 模型索引 |
| `onnx/face_morpher.onnx` | ~0.6 MB | 面部变形（ONNX） |
| `onnx/body_morpher.onnx` | ~3.9 MB | 身体变形（ONNX） |
| `onnx/merged.onnx` | ~4.4 MB | 合并模型（需 CPU 后处理） |
| **`onnx/merged_fast.onnx`** | **~4.5 MB** | **推荐：合并+GPU 后处理** |

### merged_fast.onnx 使用方法

```python
import onnxruntime, numpy as np, simplejpeg

# 1. 加载
sess = onnxruntime.InferenceSession("merged_fast.onnx",
    providers=['DmlExecutionProvider'])  # Windows GPU

# 2. 预处理角色图片（只需一次）
from PIL import Image
pil = Image.open("character.png").convert("RGBA")
img = np.array(pil, dtype=np.float32) / 255.0
# sRGB → linear
rgb = img[:,:,:3]
m = rgb <= 0.04045; rgb[m] /= 12.92; rgb[~m] = ((rgb[~m]+0.055)/1.055)**2.4
img[:,:,:3] = rgb
img[:,:,:3] *= img[:,:,3:4]    # premultiply alpha
img = img * 2 - 1              # [0,1] → [-1,1]
img_np = img.transpose(2,0,1)[None].astype(np.float32)

# 3. 推理（每帧）
pose = np.zeros((1, 45), dtype=np.float32)
# pose[18]=pose[19]=1.0  # 闭眼
# pose[26]=0.0           # 闭嘴
# pose[39]=0.5           # 头左右
# pose[42]=0.3           # 身体晃

out = sess.run(None, {"image": img_np, "pose": pose})[0]  # (1,3,512,512) uint8
jpeg = simplejpeg.encode_jpeg(out[0].transpose(1,2,0), quality=75)
```

---

## 45 个姿态参数速查

| 索引 | 参数名 | 说明 | 范围 |
|------|--------|------|------|
| 18-19 | `eye_relaxed` | 闭眼 (L/R), 1=全闭 | 0~1 |
| 26 | `mouth_aaa` | 张嘴, 0=闭嘴 | 0~1 |
| 30 | `mouth_ooo` | 圆嘴 | 0~1 |
| 36 | `mouth_smirk` | 撇嘴 | 0~1 |
| 37 | `iris_rotation_x` | 眼球上下 | -1~1 |
| 38 | `iris_rotation_y` | 眼球左右 | -1~1 |
| 39 | `head_x` | 头上下点头 | -1~1 |
| 40 | `head_y` | 头左右转 | -1~1 |
| 41 | `neck_z` | 颈部伸缩 | -1~1 |
| 42 | `body_y` | 身体左右倾 | -1~1 |
| 43 | `body_z` | 身体前后倾 | -1~1 |
| 44 | `breathing` | 呼吸幅度 | 0~1 |

完整列表参见 `src/tha4/poser/modes/pose_parameters.py`。

---

## Python 依赖摘要

如果你只需要推理（不需要训练），最小依赖：

```
onnxruntime-directml==1.17.1   # GPU 推理（Windows）
# 或 onnxruntime-gpu           # GPU 推理（需要 CUDA+cuDNN）
# 或 onnxruntime               # CPU 推理
numpy
pillow
simplejpeg                     # 快速 JPEG 编码
fastapi + uvicorn + websockets # Web 服务（可选）
```

---

## 许可证

- 代码：MIT License
- THA4 模型和图片：CC BY-NC 4.0
