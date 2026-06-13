"""
THA4 ONNX vs PyTorch side-by-side comparison.
"""
import os
import sys
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tha4.charmodel.character_model import CharacterModel
from tha4.shion.base.image_util import (
    numpy_srgb_to_linear, numpy_linear_to_srgb,
    torch_srgb_to_linear, torch_linear_to_srgb
)


MODEL_DIR = "data/character_models/lambda_00"
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
CHAR_IMG = "data/images/lambda_00.png"
OUT_DIR = "onnx_test"


def pil_to_numpy(pil_img):
    """PIL RGBA -> numpy (1,4,H,W) in [-1,1], sRGB→linear, premultiplied alpha, RGBA order."""
    img = np.array(pil_img, dtype=np.float32) / 255.0
    img[:, :, :3] = numpy_srgb_to_linear(img[:, :, :3])
    img[:, :, :3] *= img[:, :, 3:4]
    img = img * 2.0 - 1.0
    return np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)


def numpy_to_pil(arr):
    """numpy (1,4,H,W) in [-1,1] -> PIL RGBA."""
    img = arr[0].transpose(1, 2, 0)
    img = (img + 1.0) / 2.0
    alpha = img[:, :, 3]
    okay = alpha > 1e-6
    for c in range(3):
        img[:, :, c][okay] /= alpha[okay]
    img[:, :, :3] = numpy_linear_to_srgb(img[:, :, :3])
    img = np.clip(img, 0, 1) * 255
    return Image.fromarray(img.astype(np.uint8), "RGBA")


def tensor_to_pil(t):
    """PyTorch (1,4,H,W) or (4,H,W) in [-1,1] -> PIL RGBA."""
    if t.dim() == 4:
        t = t[0]
    t = (t + 1.0) / 2.0
    a = t[3]; ok = a > 1e-6
    for c in range(3):
        t[c][ok] /= a[ok]
    t[:3] = torch_linear_to_srgb(t[:3])
    t = (t * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t, "RGBA")


os.makedirs(OUT_DIR, exist_ok=True)

# ── common inputs ──────────────────────────────────────────
pil_img = Image.open(CHAR_IMG).convert("RGBA")
image_np = pil_to_numpy(pil_img)    # (1,4,512,512)

CX, CY = 256, 128 + 16              # face crop center

pose_eyes = np.zeros((1, 45), dtype=np.float32)
pose_eyes[0, 18] = 1.0             # eye_relaxed_left
pose_eyes[0, 19] = 1.0             # eye_relaxed_right

pose_zero = np.zeros((1, 45), dtype=np.float32)

# ── ONNX inference ─────────────────────────────────────────
face_onnx = ort.InferenceSession(
    os.path.join(ONNX_DIR, "face_morpher.onnx"), providers=["CPUExecutionProvider"])
body_onnx = ort.InferenceSession(
    os.path.join(ONNX_DIR, "body_morpher.onnx"), providers=["CPUExecutionProvider"])

def onnx_forward(img_np, pose_np):
    face = face_onnx.run(None, {"pose": pose_np[:, :39]})[0]
    img = img_np.copy()
    img[:, :, CY - 64:CY + 64, CX - 64:CX + 64] = face
    return body_onnx.run(None, {"image": img, "pose": pose_np})[0]

onnx_zero = onnx_forward(image_np, pose_zero)
onnx_eyes = onnx_forward(image_np, pose_eyes)

numpy_to_pil(onnx_zero).save(os.path.join(OUT_DIR, "onnx_zero.png"))
numpy_to_pil(onnx_eyes).save(os.path.join(OUT_DIR, "onnx_eyes.png"))

# ── PyTorch inference ──────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cm = CharacterModel.load(os.path.join(MODEL_DIR, "character_model.yaml"))
img_pt = cm.get_character_image(device)
pos = cm.get_poser(device)

def torch_forward(img, pose_np):
    pt_pose = torch.from_numpy(pose_np.astype(np.float32)).to(device)
    return pos.pose(img, pt_pose)

pt_zero = torch_forward(img_pt, pose_zero)
pt_eyes = torch_forward(img_pt, pose_eyes)

tensor_to_pil(pt_zero).save(os.path.join(OUT_DIR, "pytorch_zero.png"))
tensor_to_pil(pt_eyes).save(os.path.join(OUT_DIR, "pytorch_eyes.png"))

# ── difference map ─────────────────────────────────────────
def diff_map(a, b):
    d = np.abs(a - b).mean(axis=1)[0]           # (H,W)
    d = d / (d.max() + 1e-8) * 255
    return Image.fromarray(d.astype(np.uint8))

onnx_pt_zero = torch.from_numpy(onnx_zero.astype(np.float32))
onnx_pt_eyes = torch.from_numpy(onnx_eyes.astype(np.float32))

diff_map(onnx_zero, pt_zero.cpu().numpy()).save(os.path.join(OUT_DIR, "diff_zero.png"))
diff_map(onnx_eyes, pt_eyes.cpu().numpy()).save(os.path.join(OUT_DIR, "diff_eyes.png"))

# ── summary ────────────────────────────────────────────────
def stats(name, onnx, pt):
    d = np.abs(onnx - pt)
    print(f"  {name:12s}  max={d.max():.4f}  mean={d.mean():.6f}")

print("\nONNX vs PyTorch numerical differences:")
stats("zero pose", onnx_zero, pt_zero.cpu().numpy())
stats("eyes closed", onnx_eyes, pt_eyes.cpu().numpy())

print(f"\nOutputs saved to {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
