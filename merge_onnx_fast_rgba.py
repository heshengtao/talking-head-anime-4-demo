"""
Export merged_fast_rgba.onnx — GPU-side processed RGBA output.
Output: uint8 RGBA (no background burned in, preserves alpha channel).

Usage:
    python merge_onnx_fast_rgba.py <character_model_dir>
"""
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import onnx
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, interpolate

THA4_SRC = os.path.join(os.path.dirname(__file__), "src")
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


class MergedFastRGBAPipeline(Module):
    """
    Complete pipeline → uint8 RGBA with alpha channel preserved.
    Input:  image (n, 4, 512, 512) [-1,1] premultiplied
            pose  (n, 45)
    Output: rgba  (n, 4, 512, 512) uint8, sRGB, straight alpha
    """

    def __init__(self, face_morpher, body_morpher):
        super().__init__()
        self.fm_siren = face_morpher.siren
        self.fm_args = face_morpher.args
        self.bm_layers = body_morpher.siren_layers
        self.bm_last = body_morpher.last_linear
        self.bm_args = body_morpher.args

    def forward(self, image: Tensor, pose: Tensor) -> Tensor:
        n = pose.shape[0]
        device = pose.device
        dtype = torch.float32

        # ── face morpher ──
        h = w = 128
        pos = make_grid(n, h, device, dtype)
        pi = pose[:, :39].float().view(n, 39, 1, 1).repeat(1, 1, h, w)
        fm_out = self.fm_siren.forward(torch.cat([pos, pi], dim=1))

        # ── paste face ──
        cx, cy = 256, 128 + 16
        image = image.float().clone()
        image[:, :, cy - 64:cy + 64, cx - 64:cx + 64] = fm_out

        # ── body morpher ──
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

        # ── grid warp ──
        n, c, h, w = image.shape
        gc_t = torch.transpose(gc.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        bg_g = make_grid(n, h, device, dtype).permute(0, 2, 3, 1)
        g = bg_g + gc_t
        warped = grid_sample(image, g, mode='bilinear',
                             padding_mode='border', align_corners=False)
        blended = (1.0 - al) * warped + al * cc

        # ── GPU post-process → uint8 RGBA (straight alpha) ──
        out = blended.clamp(-1, 1)
        rgb = (out[:, :3] + 1.0) * 0.5
        a = (out[:, 3:4] + 1.0) * 0.5

        # un-premultiply RGB
        safe_a = torch.where(a > 1e-6, a, torch.ones_like(a))
        rgb = rgb / safe_a

        # sRGB
        rgb = torch.where(rgb <= 0.0031308, rgb * 12.92,
                          1.055 * rgb ** (1.0 / 2.4) - 0.055)

        # RGBA output: [sRGB, straight alpha] * 255
        rgba = torch.cat([rgb, a], dim=1).clamp(0, 1) * 255.0
        return rgba.to(torch.uint8)


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
        print("Usage: python merge_onnx_fast_rgba.py <character_model_dir>")
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

    merged = MergedFastRGBAPipeline(fm, bm).eval().to(device).float()
    dummy_img = torch.randn(1, 4, 512, 512, device=device, dtype=torch.float32)
    dummy_pose = torch.randn(1, 45, device=device, dtype=torch.float32)

    onnx_dir = os.path.join(model_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "merged_fast_rgba.onnx")

    print("Exporting merged_fast_rgba.onnx ...")
    torch.onnx.export(
        merged, (dummy_img, dummy_pose),
        out_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['image', 'pose'],
        output_names=['rgba'],
        dynamic_axes={
            'image': {0: 'batch'},
            'pose': {0: 'batch'},
            'rgba': {0: 'batch'},
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
    print(f"  Output: {onnx_out.shape} dtype={onnx_out.dtype}")
    print("  Export OK!")


if __name__ == "__main__":
    main()
