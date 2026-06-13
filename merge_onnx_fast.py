"""
Export merged_fast.onnx — GPU-side post-processed model.
Output: uint8 RGB directly (no CPU un-premultiply / sRGB / composite needed).

Usage:
    python merge_onnx_fast.py <character_model_dir>

Example:
    python merge_onnx_fast.py data/character_models/lambda_00
    python merge_onnx_fast.py data/distill_examples/lambda_02/character_model
"""
import sys, os, zipfile, shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import onnx
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, interpolate

THA4_SRC = os.path.join(os.path.dirname(__file__), "src")

# Green screen background for chroma-key (R, G, B)
BG_COLOR = [0, 255, 0]  # pure green #00FF00

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, THA4_SRC)

from tha4.nn.siren.face_morpher.siren_face_morpher_00 import (
    SirenFaceMorpher00, SirenFaceMorpher00Args)
from tha4.nn.siren.morpher.siren_morpher_03 import (
    SirenMorpher03, SirenMorpher03Args, SirenMorpherLevelArgs)
from tha4.nn.siren.vanilla.siren import SirenArgs


def make_grid(n, sz, device, dtype):
    h = w = sz
    ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=0).unsqueeze(0).repeat(n, 1, 1, 1)


class MergedFastPipeline(Module):
    """
    Complete pipeline: face_morpher → paste → body_morpher → post-process.
    Input:  image (n, 4, 512, 512) [-1,1] premultiplied
            pose  (n, 45)
    Output: rgb   (n, 3, 512, 512) uint8 on dark background
    """

    def __init__(self, face_morpher, body_morpher):
        super().__init__()
        self.fm_siren = face_morpher.siren
        self.fm_args = face_morpher.args
        self.bm_layers = body_morpher.siren_layers
        self.bm_last = body_morpher.last_linear
        self.bm_args = body_morpher.args
        self.register_buffer(
            'bg',
            torch.tensor(BG_COLOR, dtype=torch.float32).view(3, 1, 1) / 255)

    def forward(self, image: Tensor, pose: Tensor) -> Tensor:
        n = pose.shape[0]
        device = pose.device
        dtype = torch.float32

        # ── face morpher ────────────────────────────────
        h = w = 128
        pos = make_grid(n, h, device, dtype)
        pi = pose[:, :39].float().view(n, 39, 1, 1).repeat(1, 1, h, w)
        fm_out = self.fm_siren.forward(torch.cat([pos, pi], dim=1))

        # ── paste face into image ───────────────────────
        cx, cy = 256, 128 + 16
        image = image.float().clone()
        image[:, :, cy - 64:cy + 64, cx - 64:cx + 64] = fm_out

        # ── body morpher ────────────────────────────────
        x = None
        for i, la in enumerate(self.bm_args.level_args):
            h = w = la.image_size
            pos = make_grid(n, h, device, dtype)
            pi = pose.float().view(n, 45, 1, 1).repeat(1, 1, h, w)
            pp = torch.cat([pos, pi], dim=1)
            if i == 0:
                x = self.bm_layers[i].forward(pp)
            else:
                x = interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
                x = torch.cat([x, pp], dim=1)
                x = self.bm_layers[i].forward(x)

        s = self.bm_last(x)
        gc = s[:, 0:2, :, :]
        al = s[:, 2:3, :, :]
        cc = s[:, 3:, :, :]

        # ── grid warp ────────────────────────────────────
        n, c, h, w = image.shape
        gc_t = torch.transpose(gc.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        bg_g = make_grid(n, h, device, dtype).permute(0, 2, 3, 1)
        g = bg_g + gc_t
        warped = grid_sample(image, g, mode='bilinear',
                             padding_mode='border', align_corners=False)
        blended = (1.0 - al) * warped + al * cc

        # ── GPU post-process to uint8 RGB ────────────────
        out = blended.clamp(-1, 1)
        rgb = (out[:, :3] + 1.0) * 0.5   # [0, 1]
        a = (out[:, 3:4] + 1.0) * 0.5    # [0, 1]

        # un-premultiply
        safe_a = torch.where(a > 1e-6, a, torch.ones_like(a))
        rgb = rgb / safe_a

        # sRGB
        rgb = torch.where(rgb <= 0.0031308, rgb * 12.92,
                          1.055 * rgb ** (1.0 / 2.4) - 0.055)

        # composite on dark bg
        rgb = rgb * a + self.bg.to(device) * (1.0 - a)
        return (rgb.clamp(0, 1) * 255.0).to(torch.uint8)


# ── helpers ────────────────────────────────────────────────

def load_face_morpher(path, device):
    m = SirenFaceMorpher00(SirenFaceMorpher00Args(
        image_size=128, image_channels=4, pose_size=39,
        siren_args=SirenArgs(in_channels=41, out_channels=4,
                             intermediate_channels=128, num_sine_layers=8)))
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).float()


def load_body_morpher(path, device):
    m = SirenMorpher03(SirenMorpher03Args(
        image_size=512, image_channels=4, pose_size=45,
        level_args=[
            SirenMorpherLevelArgs(image_size=128, intermediate_channels=360, num_sine_layers=3),
            SirenMorpherLevelArgs(image_size=256, intermediate_channels=180, num_sine_layers=3),
            SirenMorpherLevelArgs(image_size=512, intermediate_channels=90, num_sine_layers=3),
        ]))
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).float()


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_onnx_fast.py <character_model_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]
    fm_path = os.path.join(model_dir, "face_morpher.pt")
    bm_path = os.path.join(model_dir, "body_morpher.pt")
    assert os.path.exists(fm_path), f"Not found: {fm_path}"
    assert os.path.exists(bm_path), f"Not found: {bm_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fm = load_face_morpher(fm_path, device)
    bm = load_body_morpher(bm_path, device)

    merged = MergedFastPipeline(fm, bm).eval().to(device).float()
    dummy_img = torch.randn(1, 4, 512, 512, device=device, dtype=torch.float32)
    dummy_pose = torch.randn(1, 45, device=device, dtype=torch.float32)

    onnx_dir = os.path.join(model_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "merged_fast.onnx")

    print("Exporting merged_fast.onnx ...")
    torch.onnx.export(
        merged, (dummy_img, dummy_pose),
        out_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['image', 'pose'],
        output_names=['rgb'],
        dynamic_axes={
            'image': {0: 'batch'},
            'pose': {0: 'batch'},
            'rgb': {0: 'batch'},
        }
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Verify
    with torch.no_grad():
        torch_out = merged(dummy_img, dummy_pose).cpu().numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {
        'image': dummy_img.cpu().numpy(),
        'pose': dummy_pose.cpu().numpy()
    })[0]
    diff = np.abs(torch_out.astype(float) - onnx_out.astype(float)).max()
    print(f"  Max diff (PyTorch vs ONNX): {diff:.1f}")
    print("  Export OK!")

    # ── Package output ────────────────────────────────────
    char_png = os.path.join(model_dir, "character.png")
    if not os.path.exists(char_png):
        # try fallback paths
        alt = os.path.join(os.path.dirname(model_dir), "..", "..", "images",
                           os.path.basename(os.path.dirname(model_dir)) + ".png")
        if os.path.exists(alt):
            char_png = alt

    if os.path.exists(char_png):
        model_name = os.path.basename(model_dir.rstrip('/\\'))
        output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, f"{model_name}.zip")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, "model.onnx")
            zf.write(char_png, "character.png")

        zip_size = os.path.getsize(zip_path) / 1024
        print(f"  Packaged: {zip_path} ({zip_size:.0f} KB)")
        print(f"  Contents: model.onnx + character.png")
    else:
        print(f"  WARNING: character.png not found, skipping ZIP packaging")


if __name__ == "__main__":
    main()
