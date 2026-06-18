"""
ONNX export optimized for Apple Silicon CoreML EP.

Strategy:
  1. Pre-compute all position grids as constants (no dynamic Shape/Range/Expand)
  2. Fixed batch=1 (no dynamic axes)
  3. Remove sRGB/alpha post-processing (output raw linear float32)
  4. Eliminates 110 unsupported nodes → target 90%+ CoreML coverage

Usage:
    python export_onnx_coreml.py <character_model_dir> [--output_dir <dir>]

Example:
    python export_onnx_coreml.py data/character_models/lambda_00
"""

import sys, os, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, interpolate
import numpy as np
import onnx

THA4_SRC = os.path.join(os.path.dirname(__file__), "src")
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, THA4_SRC)

from tha4.nn.siren.face_morpher.siren_face_morpher_00 import (
    SirenFaceMorpher00, SirenFaceMorpher00Args)
from tha4.nn.siren.morpher.siren_morpher_03 import (
    SirenMorpher03, SirenMorpher03Args, SirenMorpherLevelArgs)
from tha4.nn.siren.vanilla.siren import SirenArgs


def precompute_grid(image_size, device, dtype=torch.float32):
    """Pre-compute position grid as a constant buffer."""
    h = w = image_size
    ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([gx, gy], dim=0)  # (2, h, w)
    return grid.unsqueeze(0)  # (1, 2, h, w)


class MergedCoreMLPipeline(Module):
    """
    CoreML-optimized merged pipeline.
    Uses pre-computed position grids (registered as buffers = ONNX initializers).
    Outputs raw linear float32 (no sRGB, no uint8 cast).
    Fixed batch=1.
    """

    def __init__(self, face_morpher, body_morpher):
        super().__init__()
        self.fm_siren = face_morpher.siren
        self.fm_args = face_morpher.args
        self.bm_layers = body_morpher.siren_layers
        self.bm_last = body_morpher.last_linear
        self.bm_args = body_morpher.args

        # Pre-compute grids as buffers (become ONNX initializers = CoreML constants)
        self.register_buffer('grid_128', precompute_grid(128, torch.device('cpu')))
        self.register_buffer('grid_256', precompute_grid(256, torch.device('cpu')))
        self.register_buffer('grid_512', precompute_grid(512, torch.device('cpu')))

        # Pre-compute face paste mask (1 = use face, 0 = keep original)
        paste_mask = torch.zeros(1, 4, 512, 512, dtype=torch.float32)
        cx, cy = 256, 128 + 16
        paste_mask[:, :, cy - 64:cy + 64, cx - 64:cx + 64] = 1.0
        self.register_buffer('paste_mask', paste_mask)

    def forward(self, image: Tensor, pose: Tensor) -> Tensor:
        # ── face morpher ────────────────────────────────
        h = 128
        pos_fm = self.grid_128.to(image.device, non_blocking=True)  # (1, 2, 128, 128)
        pi_fm = pose[:, :39].float().view(1, 39, 1, 1).repeat(1, 1, h, h)
        fm_out = self.fm_siren.forward(torch.cat([pos_fm, pi_fm], dim=1))

        # ── paste face into image (pad-based, no ScatterND) ──
        # Pad: left=192, right=192, top=80, bottom=304
        face_padded = torch.nn.functional.pad(fm_out, (192, 192, 80, 304), mode='constant', value=0.0)
        pm = self.paste_mask.to(image.device, non_blocking=True)
        image = image * (1.0 - pm) + face_padded * pm

        # ── body morpher ────────────────────────────────
        grid_buffers = [self.grid_128, self.grid_256, self.grid_512]
        x = None
        for i, la in enumerate(self.bm_args.level_args):
            h = la.image_size
            pos_bm = grid_buffers[i].to(image.device, non_blocking=True)  # (1, 2, h, h)
            pi_bm = pose.float().view(1, 45, 1, 1).repeat(1, 1, h, h)
            pp = torch.cat([pos_bm, pi_bm], dim=1)

            if i == 0:
                x = self.bm_layers[i].forward(pp)
            else:
                x = interpolate(x, size=(h, h), mode='bilinear', align_corners=False)
                x = torch.cat([x, pp], dim=1)
                x = self.bm_layers[i].forward(x)

        s = self.bm_last(x)
        gc = s[:, 0:2, :, :]   # grid change
        al = s[:, 2:3, :, :]   # alpha
        cc = s[:, 3:, :, :]    # color change

        # ── grid warp ────────────────────────────────────
        n, c, h, w = image.shape
        gc_t = torch.transpose(gc.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        bg_g = self.grid_512.to(image.device, non_blocking=True).permute(0, 2, 3, 1)
        g = bg_g + gc_t
        warped = grid_sample(image, g, mode='bilinear',
                             padding_mode='border', align_corners=False)
        blended = (1.0 - al) * warped + al * cc

        # ── Output raw linear RGBA in [-1, 1] ─────────────
        # No sRGB, no uint8, no premultiply operations
        # Post-processing can be done on the host side
        return blended


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
    parser = argparse.ArgumentParser(description='Export CoreML-optimized ONNX model')
    parser.add_argument('model_dir', help='Path to character model directory')
    parser.add_argument('--output_dir', '-o', default=None,
                        help='Output directory (default: model_dir/onnx)')
    args = parser.parse_args()

    model_dir = args.model_dir
    fm_path = os.path.join(model_dir, "face_morpher.pt")
    bm_path = os.path.join(model_dir, "body_morpher.pt")
    assert os.path.exists(fm_path), f"Not found: {fm_path}"
    assert os.path.exists(bm_path), f"Not found: {bm_path}"

    out_dir = args.output_dir or os.path.join(model_dir, "onnx")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")

    fm = load_face_morpher(fm_path, device)
    bm = load_body_morpher(bm_path, device)
    print(f"Face morpher params: {sum(p.numel() for p in fm.parameters()):,}")
    print(f"Body morpher params: {sum(p.numel() for p in bm.parameters()):,}")

    merged = MergedCoreMLPipeline(fm, bm).eval().to(device).float()
    print(f"Total params: {sum(p.numel() for p in merged.parameters()):,}")
    print(f"Buffers (grids): {sum(b.numel() for b in merged.buffers()):,}")

    # Build a dummy model with explicit inputs to make ONNX export cleaner
    class ExportWrapper(Module):
        def __init__(self, pipeline):
            super().__init__()
            self.pipeline = pipeline

        def forward(self, image, pose):
            return self.pipeline.forward(image, pose)

    export_model = ExportWrapper(merged).eval().to(device).float()
    dummy_img = torch.randn(1, 4, 512, 512, device=device, dtype=torch.float32)
    dummy_pose = torch.randn(1, 45, device=device, dtype=torch.float32)

    out_path = os.path.join(out_dir, "merged_coreml.onnx")

    print("Exporting merged_coreml.onnx (fixed batch, pre-computed grids, no post-process) ...")
    torch.onnx.export(
        export_model,
        (dummy_img, dummy_pose),
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image', 'pose'],
        output_names=['blended'],
        # NO dynamic_axes → fixed batch=1
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # ── Node analysis ────────────────────────────────────
    from collections import Counter
    ops = Counter([n.op_type for n in onnx_model.graph.node])
    total = len(onnx_model.graph.node)
    print(f"\n  ONNX nodes: {total}")
    for op, count in ops.most_common(20):
        print(f"    {op:20s}: {count:4d}")

    # ── Verify numerical consistency ─────────────────────
    with torch.no_grad():
        torch_out = merged(dummy_img, dummy_pose).cpu().numpy()

    import onnxruntime as ort
    for provider in ['CPUExecutionProvider', 'CoreMLExecutionProvider']:
        try:
            if provider == 'CoreMLExecutionProvider':
                sess = ort.InferenceSession(out_path, providers=[
                    ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
                    'CPUExecutionProvider'
                ])
            else:
                sess = ort.InferenceSession(out_path, providers=[provider])

            onnx_out = sess.run(None, {
                'image': dummy_img.cpu().numpy(),
                'pose': dummy_pose.cpu().numpy()
            })[0]
            diff = np.abs(torch_out.astype(float) - onnx_out.astype(float)).max()
            print(f"  Max diff (PyTorch vs ONNX [{provider}]): {diff:.6f}")
        except Exception as e:
            print(f"  [{provider}] Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
