"""
Merge face_morpher + body_morpher into a single ONNX model.

Usage:
    python merge_onnx.py <character_model_dir>

Example:
    python merge_onnx.py data/character_models/lambda_00
    python merge_onnx.py data/distill_examples/lambda_02/character_model
"""
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, interpolate
import numpy as np
import onnx

THA4_SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, THA4_SRC)

from tha4.nn.siren.face_morpher.siren_face_morpher_00 import (
    SirenFaceMorpher00, SirenFaceMorpher00Args)
from tha4.nn.siren.morpher.siren_morpher_03 import (
    SirenMorpher03, SirenMorpher03Args, SirenMorpherLevelArgs)
from tha4.nn.siren.vanilla.siren import SirenArgs


def make_position_grid(n: int, image_size: int, device, dtype):
    h = w = image_size
    ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=0).unsqueeze(0).repeat(n, 1, 1, 1)


class MergedPipeline(Module):
    """face_morpher + face paste + body_morpher in one graph.
    Input:  image (batch, 4, 512, 512), pose (batch, 45)
    Output: blended_image (batch, 4, 512, 512)
    """
    def __init__(self, face_morpher, body_morpher):
        super().__init__()
        # ---------- face_morpher part ----------
        self.fm_siren = face_morpher.siren
        self.fm_args = face_morpher.args  # image_size=128, pose_size=39, siren_args

        # ---------- body_morpher part ----------
        self.bm_siren_layers = body_morpher.siren_layers
        self.bm_last_linear = body_morpher.last_linear
        self.bm_args = body_morpher.args

    @staticmethod
    def apply_grid_change(grid_change: Tensor, image: Tensor) -> Tensor:
        n, c, h, w = image.shape
        device = grid_change.device
        dtype = grid_change.dtype
        gc = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        bg = torch.stack([gx, gy], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        grid = bg + gc
        return grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, image: Tensor, pose: Tensor) -> Tensor:
        n = pose.shape[0]
        device = pose.device
        dtype = pose.dtype

        # === step 1: face_morpher ===
        fm_h = self.fm_args.image_size   # 128
        fm_w = fm_h
        pos = make_position_grid(n, fm_h, device, dtype)
        pi = pose[:, 0:39].view(n, 39, 1, 1).repeat(1, 1, fm_h, fm_w)
        fm_out = self.fm_siren.forward(torch.cat([pos, pi], dim=1))  # (n, 4, 128, 128)

        # === step 2: paste face into image ===
        cx, cy = 256, 128 + 16
        image = image.clone()
        image[:, :, cy - 64:cy + 64, cx - 64:cx + 64] = fm_out

        # === step 3: body_morpher ===
        x = None
        for i, la in enumerate(self.bm_args.level_args):
            h = w = la.image_size
            pos = make_position_grid(n, h, device, dtype)
            pi = pose.view(n, 45, 1, 1).repeat(1, 1, h, w)
            pp = torch.cat([pos, pi], dim=1)
            if i == 0:
                x = self.bm_siren_layers[i].forward(pp)
            else:
                x = interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
                x = torch.cat([x, pp], dim=1)
                x = self.bm_siren_layers[i].forward(x)

        s = self.bm_last_linear(x)
        gc = s[:, 0:2, :, :]
        al = s[:, 2:3, :, :]
        cc = s[:, 3:, :, :]
        wi = self.apply_grid_change(gc, image)
        return (1.0 - al) * wi + al * cc


def load_face_morpher(path, device):
    m = SirenFaceMorpher00(SirenFaceMorpher00Args(
        image_size=128, image_channels=4, pose_size=39,
        siren_args=SirenArgs(in_channels=41, out_channels=4,
                             intermediate_channels=128, num_sine_layers=8)))
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)


def load_body_morpher(path, device):
    m = SirenMorpher03(SirenMorpher03Args(
        image_size=512, image_channels=4, pose_size=45,
        level_args=[
            SirenMorpherLevelArgs(image_size=128, intermediate_channels=360, num_sine_layers=3),
            SirenMorpherLevelArgs(image_size=256, intermediate_channels=180, num_sine_layers=3),
            SirenMorpherLevelArgs(image_size=512, intermediate_channels=90, num_sine_layers=3),
        ]))
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_onnx.py <character_model_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]
    fm_path = os.path.join(model_dir, "face_morpher.pt")
    bm_path = os.path.join(model_dir, "body_morpher.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {fm_path}")
    fm = load_face_morpher(fm_path, device)
    print(f"Loading {bm_path}")
    bm = load_body_morpher(bm_path, device)

    merged = MergedPipeline(fm, bm).eval().to(device)
    dummy_img = torch.randn(1, 4, 512, 512, device=device, dtype=torch.float32)
    dummy_pose = torch.randn(1, 45, device=device, dtype=torch.float32)

    onnx_dir = os.path.join(model_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "merged.onnx")

    print("Exporting merged model ...")
    torch.onnx.export(
        merged,
        (dummy_img, dummy_pose),
        out_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['image', 'pose'],
        output_names=['blended_image'],
        dynamic_axes={
            'image': {0: 'batch'},
            'pose': {0: 'batch'},
            'blended_image': {0: 'batch'},
        }
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Saved: {out_path}")

    # Verify
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = merged(dummy_img, dummy_pose).cpu().numpy()
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {
        'image': dummy_img.cpu().numpy(),
        'pose': dummy_pose.cpu().numpy()
    })
    diff = np.abs(torch_out - onnx_out[0]).max()
    print(f"  Max diff (PyTorch vs ONNX): {diff:.6f}")

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")
    print("  Merged ONNX export OK!")


if __name__ == "__main__":
    main()
