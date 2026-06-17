"""
Bake character texture into ONNX model — eliminate separate character.png.

The result is a single-input model: pose (1,45) → rgb (1,3,512,512) uint8.

Usage:
    python bake_texture.py <merged_fast.onnx> <character.png> <output.onnx>

Example:
    python bake_texture.py merged_fast.onnx character.png merged_baked.onnx
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, numpy_helper
from PIL import Image


def _srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def preprocess_texture(path: str) -> np.ndarray:
    """Load RGBA PNG → (1, 4, 512, 512) float32 in [-1,1], premultiplied alpha."""
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    img[:, :, :3] = _srgb_to_linear(img[:, :, :3])
    img[:, :, :3] *= img[:, :, 3:4]            # premultiply alpha
    img = img * 2.0 - 1.0                       # [0,1] → [-1,1]
    img = np.expand_dims(img.transpose(2, 0, 1), 0)  # HWC → NCHW
    return img.astype(np.float32)


def bake(model_path: str, texture_path: str, output_path: str):
    """Embed preprocessed texture as a constant in the ONNX graph."""
    model = onnx.load(model_path)
    graph = model.graph
    onnx.checker.check_model(model)

    # 1. Preprocess the texture
    texture = preprocess_texture(texture_path)  # (1, 4, 512, 512) float32

    # 2. Create a Constant initializer for the image tensor
    image_constant = numpy_helper.from_array(texture, name="baked_image")

    # 3. Find all nodes that consume the 'image' input
    image_input_name = None
    for inp in graph.input:
        if inp.name == "image":
            image_input_name = inp.name
            break

    if image_input_name is None:
        raise RuntimeError("Model has no 'image' input")

    # 4. Replace 'image' references in all nodes with the constant
    #    We need to add the initializer and rewire node inputs
    graph.initializer.append(image_constant)

    # 5. Rewire: for every node that takes 'image' as input, replace with 'baked_image'
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == "image":
                node.input[i] = "baked_image"

    # 6. Remove 'image' from graph inputs (keep only 'pose')
    new_inputs = [inp for inp in graph.input if inp.name != "image"]
    del graph.input[:]
    graph.input.extend(new_inputs)

    # 7. Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[bake] Texture embedded → {output_path} ({size_kb:.0f} KB)")

    # Show I/O
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in graph.output]
    print(f"[bake] Input  → {inputs}")
    print(f"[bake] Output → {outputs}")

    # Quick functional test
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    pose_zero = np.zeros((1, 45), dtype=np.float32)
    out = sess.run(None, {"pose": pose_zero})[0]
    print(f"[bake] Test   → output shape {out.shape}, dtype {out.dtype}")
    print("[bake] Done!")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python bake_texture.py <merged_fast.onnx> <character.png> <output.onnx>")
        sys.exit(1)

    bake(sys.argv[1], sys.argv[2], sys.argv[3])
