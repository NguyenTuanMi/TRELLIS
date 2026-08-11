"""
TRELLIS inference server.

Runs on your remote GPU machine (inside the `trellis` conda env).
Exposes a single endpoint that accepts multiple images and returns a .glb file.

Start it with:
    conda activate trellis
    pip install fastapi "uvicorn[standard]" python-multipart
    python inference_server.py --host 0.0.0.0 --port 8000

Then, from your laptop, either:
  - open the port on your firewall / cloud security group, or
  - SSH-tunnel it:  ssh -L 8000:localhost:8000 user@remote-server
"""

import argparse
import io
import os
import shutil
import tempfile
import uuid

os.environ.setdefault("SPCONV_ALGO", "native")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

app = FastAPI(title="TRELLIS Inference Server")

# Allow the local client (served from file:// or localhost) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across requests (loading the pipeline is slow).
pipeline = None

# Where finished .glb files are written, keyed by job id, so the client can fetch them.
OUTPUT_DIR = tempfile.mkdtemp(prefix="trellis_outputs_")


def get_pipeline():
    global pipeline
    if pipeline is None:
        from trellis.pipelines import TrellisImageTo3DPipeline

        print("[server] loading TRELLIS pipeline (first request only)...")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")

        pipeline.cuda()
        print("[server] pipeline ready.")
    return pipeline


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/generate")
async def generate(
    images: list[UploadFile] = File(...),
    multiimage_algo: str = "stochastic",  # "stochastic" or "multidiffusion"
):
    os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
    os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
    if not images:
        raise HTTPException(400, "No images uploaded.")

    pil_images = []
    for f in images:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception as e:
            raise HTTPException(400, f"Could not read image '{f.filename}': {e}")
        pil_images.append(img)

    pipe = get_pipeline()

    from trellis.utils import postprocessing_utils

    print(f"[server] running pipeline on {len(pil_images)} images "
          f"(algo={multiimage_algo})...")

    if len(pil_images) == 1:
        outputs = pipe.run(pil_images[0], seed=1)
    else:
        outputs = pipe.run_multi_image(
            pil_images,
            seed=1,
            sparse_structure_sampler_params={
                    "steps": 12,
                    "cfg_strength": 7.5,
            },
            slat_sampler_params={
                    "steps": 12,
                    "cfg_strength": 3,
            },
        )

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )

    job_id = str(uuid.uuid4())
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")
    glb.export(out_path)

    print(f"[server] done -> {out_path}")
    return {"job_id": job_id, "download_url": f"/download/{job_id}"}


@app.get("/download/{job_id}")
def download(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found (server may have restarted).")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
