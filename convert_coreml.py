"""
Convert THA4 model directly to CoreML .mlpackage with baked texture.

Output: single-input model (pose only) → blended image.
Texture is preprocessed and embedded as a constant in the model.

Usage:
    python convert_coreml.py <character_model_dir>

Example:
    python convert_coreml.py data/character_models/lambda_00
"""

import sys, os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import coremltools as ct
from torch.nn.functional import grid_sample, interpolate
from PIL import Image

THA4_SRC = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, THA4_SRC)

from tha4.nn.siren.face_morpher.siren_face_morpher_00 import (
    SirenFaceMorpher00, SirenFaceMorpher00Args)
from tha4.nn.siren.morpher.siren_morpher_03 import (
    SirenMorpher03, SirenMorpher03Args, SirenMorpherLevelArgs)
from tha4.nn.siren.vanilla.siren import SirenArgs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def precompute_grid(image_size, device, dtype=torch.float32):
    h = w = image_size
    ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=0).unsqueeze(0)


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def preprocess_texture(path):
    """Load RGBA PNG → (1, 4, 512, 512) float32 in [-1,1], premultiplied alpha."""
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    img[:, :, :3] = srgb_to_linear(img[:, :, :3])
    img[:, :, :3] *= img[:, :, 3:4]
    img = img * 2.0 - 1.0
    img = np.expand_dims(img.transpose(2, 0, 1), 0)
    return torch.from_numpy(img.astype(np.float32))


class CoreMLBakedPipeline(torch.nn.Module):
    """Single-input pipeline: pose → blended image. Texture baked as constant."""

    def __init__(self, fm, bm, texture_path):
        super().__init__()
        self.fm_siren = fm.siren
        self.bm_layers = bm.siren_layers
        self.bm_last = bm.last_linear
        self.register_buffer('grid_128', precompute_grid(128, torch.device('cpu')))
        self.register_buffer('grid_256', precompute_grid(256, torch.device('cpu')))
        self.register_buffer('grid_512', precompute_grid(512, torch.device('cpu')))

        paste_mask = torch.zeros(1, 4, 512, 512)
        cx, cy = 256, 128 + 16
        paste_mask[:, :, cy - 64:cy + 64, cx - 64:cx + 64] = 1.0
        self.register_buffer('paste_mask', paste_mask)

        # Bake texture into model
        texture = preprocess_texture(texture_path)
        self.register_buffer('baked_texture', texture)

    def forward(self, pose):
        image = self.baked_texture.clone()

        # Face morpher
        pi_fm = pose[:, :39].view(1, 39, 1, 1).repeat(1, 1, 128, 128)
        fm_out = self.fm_siren(torch.cat([self.grid_128, pi_fm], dim=1))

        # Paste face (pad-based)
        face_padded = torch.nn.functional.pad(fm_out, (192, 192, 80, 304), 'constant', 0.0)
        image = image * (1.0 - self.paste_mask) + face_padded * self.paste_mask

        # Body morpher (3 levels)
        grid_bufs = [self.grid_128, self.grid_256, self.grid_512]
        sizes = [128, 256, 512]
        x = None
        for i in range(3):
            sz = sizes[i]
            pos = grid_bufs[i]
            pi = pose.view(1, 45, 1, 1).repeat(1, 1, sz, sz)
            pp = torch.cat([pos, pi], dim=1)
            if i == 0:
                x = self.bm_layers[i](pp)
            else:
                x = interpolate(x, size=(sz, sz), mode='bilinear', align_corners=False)
                x = torch.cat([x, pp], dim=1)
                x = self.bm_layers[i](x)

        s = self.bm_last(x)
        gc = s[:, 0:2, :, :]
        al = s[:, 2:3, :, :]
        cc = s[:, 3:, :, :]

        # Grid warp
        gc_t = torch.transpose(gc.view(1, 2, 512 * 512), 1, 2).view(1, 512, 512, 2)
        bg_g = self.grid_512.permute(0, 2, 3, 1)
        g = bg_g + gc_t
        warped = grid_sample(image, g, mode='bilinear', padding_mode='border', align_corners=False)
        return (1.0 - al) * warped + al * cc


def load_models(model_dir):
    fm_path = os.path.join(model_dir, "face_morpher.pt")
    bm_path = os.path.join(model_dir, "body_morpher.pt")

    fm = SirenFaceMorpher00(SirenFaceMorpher00Args(
        image_size=128, image_channels=4, pose_size=39,
        siren_args=SirenArgs(in_channels=41, out_channels=4,
                             intermediate_channels=128, num_sine_layers=8)))
    fm.load_state_dict(torch.load(fm_path, map_location='cpu', weights_only=True))
    fm = fm.float().eval()

    bm = SirenMorpher03(SirenMorpher03Args(
        image_size=512, image_channels=4, pose_size=45,
        level_args=[
            SirenMorpherLevelArgs(image_size=128, intermediate_channels=360, num_sine_layers=3),
            SirenMorpherLevelArgs(image_size=256, intermediate_channels=180, num_sine_layers=3),
            SirenMorpherLevelArgs(image_size=512, intermediate_channels=90, num_sine_layers=3),
        ]))
    bm.load_state_dict(torch.load(bm_path, map_location='cpu', weights_only=True))
    bm = bm.float().eval()
    return fm, bm


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "data/character_models/lambda_00"
    model_dir = os.path.join(os.path.dirname(__file__), model_dir)

    char_png = os.path.join(model_dir, "character.png")
    assert os.path.exists(char_png), f"Texture not found: {char_png}"

    print("Loading models...")
    fm, bm = load_models(model_dir)
    model = CoreMLBakedPipeline(fm, bm, char_png).eval()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Buffers (grids + texture): {sum(b.numel() for b in model.buffers()):,}")

    dummy_pose = torch.randn(1, 45)

    # Test PyTorch forward
    print("Testing PyTorch forward (pose only)...")
    torch_out = model(dummy_pose)
    print(f"  Output shape: {torch_out.shape}")

    # JIT trace
    print("Tracing with torch.jit.trace...")
    traced = torch.jit.trace(model, dummy_pose)
    print("  Trace OK")

    # Convert to CoreML
    print("Converting to CoreML mlprogram (single-input: pose)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, 45), name="pose")],
        outputs=[ct.TensorType(name="blended")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    out_path = os.path.join(PROJECT_ROOT, "output", "model_coreml.mlpackage")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlmodel.save(out_path)
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, files in os.walk(out_path) for f in files) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Verify
    print("Verifying numerical accuracy...")
    result = mlmodel.predict({'pose': dummy_pose.numpy()})
    out_key = [k for k in result if k != 'pose'][0]
    coreml_out = result[out_key]
    torch_np = torch_out.detach().numpy()
    diff = np.abs(coreml_out - torch_np).max()
    mean_diff = np.abs(coreml_out - torch_np).mean()
    print(f"  Max diff: {diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Benchmark
    print("\nBenchmarking (pose only → blended image)...")
    for _ in range(5):
        mlmodel.predict({'pose': dummy_pose.numpy()})
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        mlmodel.predict({'pose': dummy_pose.numpy()})
    t = time.perf_counter() - t0
    print(f"  CoreML (ANE): {t/N*1000:.1f} ms/frame, {N/t:.1f} fps")

    # Test with real poses
    print("\nTesting with default pose...")
    real_pose = np.zeros((1, 45), dtype=np.float32)
    real_pose[0, 18] = 1.0  # eyes open
    real_pose[0, 19] = 1.0

    real_out = mlmodel.predict({'pose': real_pose})[out_key]
    print(f"  Output shape: {real_out.shape}, range: [{real_out.min():.3f}, {real_out.max():.3f}]")

    # Save preview
    preview = np.clip((real_out[0, :3] + 1.0) / 2.0, 0, 1).transpose(1, 2, 0)
    preview = (preview * 255).astype(np.uint8)
    preview_path = os.path.join(os.path.dirname(out_path), "coreml_baked_preview.png")
    Image.fromarray(preview).save(preview_path)
    print(f"  Preview saved to {preview_path}")

    print("\nDone! Single-input model saved.")


if __name__ == "__main__":
    main()
