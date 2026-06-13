"""
THA4 Web Demo — auto-select best backend:
  1. ONNX CUDA GPU (if cuDNN 9 installed)
  2. PyTorch CUDA GPU (fallback)
Usage: python web_demo/server.py
"""
import asyncio, json, math, os, sys, time, io
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

CHAR_MODEL_DIR = os.environ.get("THA4_CHAR_MODEL",
    os.path.join(PROJECT_ROOT, "data", "distill_examples", "lambda_02", "character_model"))
CHAR_MODEL_YAML = os.path.join(CHAR_MODEL_DIR, "character_model.yaml")
if not os.path.exists(CHAR_MODEL_YAML):
    CHAR_MODEL_YAML = os.path.join(PROJECT_ROOT, "data", "character_models", "lambda_00", "character_model.yaml")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

_BG = np.array([26, 26, 46], dtype=np.float32) / 255.0


def _srgb_to_linear(x):
    x=np.clip(x,0,1); return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def _linear_to_srgb(x):
    x=np.clip(x,0,1); return np.where(x<=0.003130804953560372,x*12.92,1.055*(x**(1/2.4))-0.055)


class IdlePoseGenerator:
    def __init__(self):
        self.t=0.0; self.last=time.perf_counter()
        self._pby=np.random.random()*math.pi*2; self._pbz=np.random.random()*math.pi*2
        self._pbr=np.random.random()*math.pi*2; self._phx=np.random.random()*math.pi*2
        self._phy=np.random.random()*math.pi*2
        self._bsy=0.6+np.random.random()*0.4; self._bsz=0.4+np.random.random()*0.3
        self._bsb=1.6+np.random.random()*0.4
        self.next_blink=0.6+np.random.random()*1.0; self.blink_state=0; self.blink_timer=0.0
        self.blink_dur=0.06; self.blink_hold=0.08
        # mouse tracking — smoothed
        self.mx=0.0; self.my=0.0
        self._mouse_x=0.0; self._mouse_y=0.0
    def _rb(self): return 2.0+np.random.random()*4.0
    def step(self):
        now=time.perf_counter(); dt=now-self.last; self.last=now; self.t+=dt
        p=np.zeros(45,dtype=np.float32)
        # ── idle animation ──
        idle_breath=0.40*math.sin(self.t*self._bsb+self._pbr)
        idle_body_y=0.35*math.sin(self.t*self._bsy+self._pby)
        idle_body_z=0.30*math.sin(self.t*self._bsz+self._pbz)
        idle_head_x=0.18*math.sin(self.t*1.1+self._phx)
        idle_head_y=0.14*math.sin(self.t*1.3+self._phy)
        idle_neck=0.08*math.sin(self.t*0.55)
        idle_iris_x=0.10*math.sin(self.t*0.45+self._phy)
        idle_iris_y=0.07*math.sin(self.t*0.55+self._phx)
        # ── mouse tracking (smoothed toward target) ──
        self.mx += (self._mouse_x - self.mx) * min(dt*8.0, 1.0)
        self.my += (self._mouse_y - self.my) * min(dt*8.0, 1.0)
        mx, my = self.mx, self.my
        # ── blend ──
        p[44] = idle_breath
        p[42] = idle_body_y + mx * 0.20              # body lean (increased)
        p[43] = idle_body_z + my * 0.15
        p[39] = idle_head_x - my * 0.85              # head tilt (increased)
        p[40] = idle_head_y - mx * 0.90              # head turn (increased)
        p[41] = idle_neck
        p[37] = idle_iris_x - my * 0.85              # iris vertical (negated)
        p[38] = idle_iris_y - mx * 0.95              # iris horizontal (negated)
        # ── blinking ──
        self.blink_timer+=dt
        if self.blink_state==0:
            if self.blink_timer>=self.next_blink: self.blink_state=1; self.blink_timer=0.0; self.next_blink=self._rb()
        elif self.blink_state==1:
            v=min(self.blink_timer/self.blink_dur,1.0); p[18]=p[19]=v
            if v>=1.0: self.blink_state=2; self.blink_timer=0.0
        elif self.blink_state==2:
            p[18]=p[19]=1.0
            if self.blink_timer>=self.blink_hold: self.blink_state=3; self.blink_timer=0.0
        elif self.blink_state==3:
            v=1.0-min(self.blink_timer/self.blink_dur,1.0); p[18]=p[19]=v
            if v<=0.0: self.blink_state=0; self.blink_timer=0.0
        p[26]=0.0
        return p


# ── backend selection ──────────────────────────────────────
backend_info = ""

def init_pytorch():
    import torch
    from tha4.charmodel.character_model import CharacterModel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU"
    print(f"[init] PyTorch GPU: {name}")
    cm=CharacterModel.load(CHAR_MODEL_YAML)
    poser=cm.get_poser(dev); char=cm.get_character_image(dev)
    bg=torch.tensor([26,26,46],dtype=torch.float32,device=dev).view(3,1,1)/255
    def render(pose_np):
        p=torch.from_numpy(pose_np.reshape(1,45).astype(np.float32)).to(dev)
        with torch.no_grad():
            out=poser.pose(char,p)[0].clamp(-1,1)
            rgb=(out[:3]+1)/2; a=(out[3]+1)/2; m=a>1e-6; rgb[:,m]/=a[m]
            rgb=torch.where(rgb<=0.0031308,rgb*12.92,1.055*rgb**(1/2.4)-0.055)
            rgb=rgb*a.unsqueeze(0)+bg*(1-a.unsqueeze(0)); rgb=rgb.clamp(0,1)*255
            arr=rgb.to(torch.uint8).permute(1,2,0).cpu().numpy()
        buf=io.BytesIO(); Image.fromarray(arr,"RGB").save(buf,format="JPEG",quality=78,optimize=True)
        return buf.getvalue()
    return render


def init_onnx():
    import onnxruntime as ort
    onnx_dir=os.path.join(CHAR_MODEL_DIR,"onnx")
    merged=os.path.join(onnx_dir,"merged.onnx")
    if not os.path.exists(merged):
        merged=os.path.join(PROJECT_ROOT,"data","character_models","lambda_00","onnx","merged.onnx")
    char_png=os.path.join(CHAR_MODEL_DIR,"character.png")
    if not os.path.exists(char_png):
        char_png=os.path.join(PROJECT_ROOT,"data","images","lambda_00.png")
    providers=["CUDAExecutionProvider","CPUExecutionProvider"]
    sess=ort.InferenceSession(merged,providers=providers)
    actual=sess.get_providers()[0]
    if "CPU" in actual:
        raise RuntimeError(f"ONNX CUDA not available (got {actual}). Install cuDNN: pip install nvidia-cudnn-cu12")
    print(f"[init] ONNX GPU: {actual}")
    img=np.array(Image.open(char_png).convert("RGBA"),dtype=np.float32)/255.0
    img[:,:,:3]=_srgb_to_linear(img[:,:,:3]); img[:,:,:3]*=img[:,:,3:4]
    img=img*2-1; img_np=np.expand_dims(img.transpose(2,0,1),0).astype(np.float32)
    def render(pose_np):
        p=pose_np.reshape(1,45).astype(np.float32)
        out=sess.run(None,{"image":img_np,"pose":p})[0]
        img=out[0].transpose(1,2,0); img=(img+1)/2
        a=img[:,:,3:4]; rgb=img[:,:,:3]
        mk=a[:,:,0]>1e-6
        for c in range(3): rgb[:,:,c][mk]/=a[:,:,0][mk]
        rgb=_linear_to_srgb(rgb); rgb=rgb*a+_BG.reshape(1,1,3)*(1-a)
        rgb=np.clip(rgb,0,1)*255
        buf=io.BytesIO(); Image.fromarray(rgb.astype(np.uint8),"RGB").save(buf,format="JPEG",quality=78,optimize=True)
        return buf.getvalue()
    return render


render_fn = None

def init_model():
    global render_fn
    try:
        render_fn = init_onnx()
        return
    except Exception as e:
        print(f"[init] ONNX GPU failed: {e}")
    try:
        render_fn = init_pytorch()
        return
    except Exception as e:
        print(f"[init] PyTorch failed: {e}")
    raise RuntimeError("No backend available")


# ── FastAPI ────────────────────────────────────────────────
app = FastAPI(title="THA4 Web Demo")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    from pathlib import Path
    return HTMLResponse(Path(os.path.join(STATIC_DIR,"index.html")).read_text("utf-8"))

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); gen=IdlePoseGenerator()
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
            jpeg=render_fn(gen.step())
            await ws.send_bytes(jpeg)
            await recv_mouse()           # non-blocking mouse update
            await asyncio.sleep(0)
    except WebSocketDisconnect: pass


if __name__ == "__main__":
    import uvicorn
    init_model()
    print(f"[init] Ready.  http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
