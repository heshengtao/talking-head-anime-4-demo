"""
THA4 Web Demo — Mac (Apple Silicon) edition.
Auto-selects best backend:
  1. CoreML .mlpackage (M4 Neural Engine, ~74 fps)
  2. ONNX CPU (fallback, ~4 fps)

Usage: python web_demo_mac/server.py

    Open http://localhost:8000  — move mouse over canvas to control head
"""

import asyncio, json, math, os, sys, time, io

import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── Background color ──────────────────────────────────────
_BG = np.array([26, 26, 46], dtype=np.float32) / 255.0


# ── Color helpers ─────────────────────────────────────────

def _srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


# ── Idle pose generator (same as original web_demo) ──────

class IdlePoseGenerator:
    def __init__(self):
        self.t = 0.0
        self.last = time.perf_counter()
        self._pby = np.random.random() * math.pi * 2
        self._pbz = np.random.random() * math.pi * 2
        self._pbr = np.random.random() * math.pi * 2
        self._phx = np.random.random() * math.pi * 2
        self._phy = np.random.random() * math.pi * 2
        self._bsy = 0.6 + np.random.random() * 0.4
        self._bsz = 0.4 + np.random.random() * 0.3
        self._bsb = 1.6 + np.random.random() * 0.4
        self.next_blink = 0.6 + np.random.random() * 1.0
        self.blink_state = 0
        self.blink_timer = 0.0
        self.blink_dur = 0.06
        self.blink_hold = 0.08
        self.mx = 0.0
        self.my = 0.0
        self._mouse_x = 0.0
        self._mouse_y = 0.0

    def _rb(self):
        return 2.0 + np.random.random() * 4.0

    def step(self):
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        self.t += dt
        p = np.zeros(45, dtype=np.float32)
        idle_breath = 0.40 * math.sin(self.t * self._bsb + self._pbr)
        idle_body_y = 0.35 * math.sin(self.t * self._bsy + self._pby)
        idle_body_z = 0.30 * math.sin(self.t * self._bsz + self._pbz)
        idle_head_x = 0.18 * math.sin(self.t * 1.1 + self._phx)
        idle_head_y = 0.14 * math.sin(self.t * 1.3 + self._phy)
        idle_neck = 0.08 * math.sin(self.t * 0.55)
        idle_iris_x = 0.10 * math.sin(self.t * 0.45 + self._phy)
        idle_iris_y = 0.07 * math.sin(self.t * 0.55 + self._phx)
        self.mx += (self._mouse_x - self.mx) * min(dt * 8.0, 1.0)
        self.my += (self._mouse_y - self.my) * min(dt * 8.0, 1.0)
        mx, my = self.mx, self.my
        p[44] = idle_breath
        p[42] = idle_body_y + mx * 0.20
        p[43] = idle_body_z + my * 0.15
        p[39] = idle_head_x - my * 0.85
        p[40] = idle_head_y - mx * 0.90
        p[41] = idle_neck
        p[37] = idle_iris_x - my * 0.85
        p[38] = idle_iris_y - mx * 0.95
        self.blink_timer += dt
        if self.blink_state == 0:
            if self.blink_timer >= self.next_blink:
                self.blink_state = 1
                self.blink_timer = 0.0
                self.next_blink = self._rb()
        elif self.blink_state == 1:
            v = min(self.blink_timer / self.blink_dur, 1.0)
            p[18] = p[19] = v
            if v >= 1.0:
                self.blink_state = 2
                self.blink_timer = 0.0
        elif self.blink_state == 2:
            p[18] = p[19] = 1.0
            if self.blink_timer >= self.blink_hold:
                self.blink_state = 3
                self.blink_timer = 0.0
        elif self.blink_state == 3:
            v = 1.0 - min(self.blink_timer / self.blink_dur, 1.0)
            p[18] = p[19] = v
            if v <= 0.0:
                self.blink_state = 0
                self.blink_timer = 0.0
        p[26] = 0.0
        return p


# ── CoreML backend (baked single-input model) ─────────────

def find_coreml_model():
    """Search for model_coreml.mlpackage in common locations."""
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "model_coreml.mlpackage"),
        os.path.join(PROJECT_ROOT, "Lyra", "model_baked.mlpackage"),
        os.environ.get("THA4_COREML_MODEL", ""),
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def init_coreml():
    model_path = find_coreml_model()
    if not model_path:
        raise FileNotFoundError("No .mlpackage found (Lyra/model_baked.mlpackage or THA4_COREML_MODEL)")

    from coremltools.models import MLModel
    model = MLModel(model_path)
    model_info = f"CoreML — {os.path.basename(model_path)}"
    # Warmup
    _ = model.predict({"pose": np.zeros((1, 45), dtype=np.float32)})
    print(f"[init] CoreML: {model_path} — M4 Neural Engine")

    def render(pose_np):
        p = pose_np.reshape(1, 45).astype(np.float32)
        result = model.predict({"pose": p})
        key = [k for k in result if k != "pose"][0]
        blended = result[key]  # (1, 4, 512, 512) float32 [-1, 1]
        img = blended[0].transpose(1, 2, 0)
        img = (img + 1) / 2
        a = img[:, :, 3:4]
        rgb = img[:, :, :3]
        mk = a[:, :, 0] > 1e-6
        for c in range(3):
            rgb[:, :, c][mk] /= a[:, :, 0][mk]
        rgb = _linear_to_srgb(rgb)
        rgb = rgb * a + _BG.reshape(1, 1, 3) * (1 - a)
        rgb = np.clip(rgb, 0, 1) * 255
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8), "RGB").save(buf, format="JPEG", quality=78, optimize=True)
        return buf.getvalue()

    return render, model_info


# ── ONNX CPU fallback ─────────────────────────────────────

def init_onnx():
    import onnxruntime as ort

    # Try merged.onnx first, then model.onnx
    onnx_candidates = [
        os.path.join(PROJECT_ROOT, "Lyra", "model.onnx"),
        os.path.join(PROJECT_ROOT, "data", "character_models", "lambda_00", "onnx", "merged_fast.onnx"),
        os.path.join(PROJECT_ROOT, "data", "character_models", "lambda_00", "onnx", "face_morpher.onnx"),
    ]
    merged = None
    for p in onnx_candidates:
        if os.path.exists(p):
            merged = p
            break
    if not merged:
        raise FileNotFoundError("No ONNX model found")

    char_png = os.path.join(PROJECT_ROOT, "Lyra", "character.png")
    if not os.path.exists(char_png):
        char_png = os.path.join(PROJECT_ROOT, "data", "images", "lambda_00.png")

    sess = ort.InferenceSession(merged, providers=["CPUExecutionProvider"])
    print(f"[init] ONNX CPU: {merged}")

    # Preprocess character image
    img = np.array(Image.open(char_png).convert("RGBA"), dtype=np.float32) / 255.0
    img[:, :, :3] = _srgb_to_linear(img[:, :, :3])
    img[:, :, :3] *= img[:, :, 3:4]
    img = img * 2 - 1
    img_np = np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)

    model_info = f"ONNX CPU — {os.path.basename(merged)}"

    def render(pose_np):
        p = pose_np.reshape(1, 45).astype(np.float32)
        out = sess.run(None, {"image": img_np, "pose": p})[0]
        img = out[0].transpose(1, 2, 0)
        img = (img + 1) / 2
        a = img[:, :, 3:4]
        rgb = img[:, :, :3]
        mk = a[:, :, 0] > 1e-6
        for c in range(3):
            rgb[:, :, c][mk] /= a[:, :, 0][mk]
        rgb = _linear_to_srgb(rgb)
        rgb = rgb * a + _BG.reshape(1, 1, 3) * (1 - a)
        rgb = np.clip(rgb, 0, 1) * 255
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8), "RGB").save(buf, format="JPEG", quality=78, optimize=True)
        return buf.getvalue()

    return render, model_info


# ── Backend selection ─────────────────────────────────────

render_fn = None
backend_info = ""

def init_model():
    global render_fn, backend_info

    # 1. Try CoreML .mlpackage (native Mac, M4 ANE)
    try:
        render_fn, backend_info = init_coreml()
        return
    except Exception as e:
        print(f"[init] CoreML failed: {e}")

    # 2. Fall back to ONNX CPU
    try:
        render_fn, backend_info = init_onnx()
        return
    except Exception as e:
        print(f"[init] ONNX failed: {e}")

    raise RuntimeError("No backend available. Put model_baked.mlpackage in Lyra/ or model.onnx.")


# ── FastAPI ────────────────────────────────────────────────

app = FastAPI(title="THA4 Web Demo — Mac")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    from pathlib import Path
    html = Path(os.path.join(STATIC_DIR, "index.html")).read_text("utf-8")
    # Inject backend info
    html = html.replace("</body>",
        f'<script>document.getElementById("info").textContent="{backend_info}"</script></body>')
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    gen = IdlePoseGenerator()

    async def recv_mouse():
        try:
            msg = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
            data = json.loads(msg)
            gen._mouse_x = float(data.get("x", 0))
            gen._mouse_y = float(data.get("y", 0))
        except (asyncio.TimeoutError, ValueError, KeyError):
            pass
        except WebSocketDisconnect:
            raise

    try:
        while True:
            jpeg = render_fn(gen.step())
            await ws.send_bytes(jpeg)
            await recv_mouse()
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    init_model()
    print(f"[init] Ready.  http://localhost:8000  |  backend: {backend_info}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
