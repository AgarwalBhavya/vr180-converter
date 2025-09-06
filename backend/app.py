# import os, shutil, uuid, subprocess
# from fastapi import FastAPI, File, UploadFile, BackgroundTasks
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from status_store import read_status, write_status
# from processor import run_pipeline, update_status
# from pathlib import Path
#
# ROOT = Path(__file__).resolve().parent
# OUTPUTS = ROOT.parent / 'outputs'
# os.makedirs(OUTPUTS, exist_ok=True)
#
# app = FastAPI()
# app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")
#
# @app.post("/upload")
# async def upload_video(file: UploadFile = File(...)):
#     job_id = uuid.uuid4().hex[:12]
#     job_dir = OUTPUTS / job_id
#     job_dir.mkdir(parents=True, exist_ok=True)
#     in_path = job_dir / "input.mp4"
#     # save uploaded file
#     with open(in_path, "wb") as f:
#         content = await file.read()
#         f.write(content)
#     # write initial status
#     write_status(str(job_dir), {'stage': 'queued', 'progress': 0.01})
#     # spawn a background process that runs the processor script
#     # Use subprocess to isolate memory and GPU usage
#     python_bin = shutil.which("python3") or shutil.which("python") or "python"
#     proc_cmd = f'{python_bin} -u -c "from processor import run_pipeline; run_pipeline(r\'{job_dir}\', r\'{in_path}\')"'
#     # use Popen and detach so FastAPI returns quickly
#     subprocess.Popen(proc_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return {"job_id": job_id}
#
# @app.get("/status")
# def status(job_id: str):
#     job_dir = OUTPUTS / job_id
#     if not job_dir.exists():
#         return JSONResponse({"error":"job not found"}, status_code=404)
#     s = read_status(str(job_dir))
#     return s
#
# @app.get("/download")
# def download(job_id: str):
#     job_dir = OUTPUTS / job_id
#     f = job_dir / 'vr180_sbs.mp4'
#     if not f.exists():
#         return JSONResponse({"error":"not ready"}, status_code=404)
#     return FileResponse(str(f), media_type='video/mp4', filename='vr180_sbs.mp4')
#
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pathlib import Path
#
# # allow frontend served from same server (simple for demo)
# app.add_middleware(
#   CORSMiddleware,
#   allow_origins=["*"],
#   allow_credentials=True,
#   allow_methods=["*"],
#   allow_headers=["*"],
# )
#
# # serve frontend at root
# ROOT = Path(__file__).resolve().parent
# app.mount("/", StaticFiles(directory=str(ROOT.parent / 'frontend'), html=True), name="frontend")
# # keep the existing mount for outputs too
# app.mount("/outputs", StaticFiles(directory=str(ROOT.parent / 'outputs')), name="outputs")
import sys


import os
import shutil
import uuid
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from status_store import read_status, write_status

# --- paths ---
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'
OUTPUTS = PROJECT_ROOT / 'outputs'
os.makedirs(OUTPUTS, exist_ok=True)

# --- app ---
app = FastAPI()

# allow CORS for local demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve outputs directory at /outputs for downloads/previews
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")

# serve frontend assets under /static (not required by index.html, but useful for other assets)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --- helper to read/write status ---
def make_job_dir(job_id: str) -> Path:
    job_dir = OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


# --- frontend routes (keeps existing index.html references working) ---
@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Some of the frontend files refer to `/app.js` and `/style.css` directly.
# Provide explicit routes so we don't need to change the HTML.
@app.get("/app.js", response_class=FileResponse)
async def app_js():
    return FileResponse(str(FRONTEND_DIR / "app.js"))


@app.get("/style.css", response_class=FileResponse)
async def style_css():
    return FileResponse(str(FRONTEND_DIR / "style.css"))


# --- API endpoints ---
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Save uploaded file, create a job folder, write a queued status, then spawn the
    processor in a detached subprocess while capturing stdout/stderr to worker.log.
    The spawned worker uses sys.executable so it runs with the same venv/python
    as the FastAPI server (avoids ModuleNotFoundError for torch).
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = make_job_dir(job_id)
    in_path = job_dir / "input.mp4"

    # save uploaded file
    with open(in_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # initial status
    write_status(str(job_dir), {'stage': 'queued', 'progress': 0.01})

    # Use same Python executable as the running server so the worker sees the same venv
    python_bin = sys.executable
    backend_dir = str(ROOT)  # ensure processor can be imported

    # Inline script: print diagnostics (which python, torch availability), then run pipeline
    # Use explicit newline escapes so the inline script stays readable.
    inline_script = (
        "import sys, traceback\n"
        "print('WORKER: sys.executable=' + sys.executable)\n"
        "try:\n"
        "    import torch\n"
        "    print('WORKER: torch=' + torch.__version__ + ', cuda_available=' + str(torch.cuda.is_available()))\n"
        "except Exception:\n"
        "    print('WORKER: torch import failed:', file=sys.stderr)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(2)\n"
        f"sys.path.insert(0, r'{backend_dir}')\n"
        f"from processor import run_pipeline\n"
        f"run_pipeline(r'{str(job_dir)}', r'{str(in_path)}')\n"
    )

    log_path = job_dir / "worker.log"
    # spawn worker; write stdout/stderr to worker.log
    with open(log_path, "ab") as lf:
        subprocess.Popen([python_bin, "-u", "-c", inline_script], stdout=lf, stderr=lf)

    return {"job_id": job_id}

# @app.post("/upload")
# async def upload_video(file: UploadFile = File(...)):
#     """
#     Save uploaded file, create a job folder, write a queued status, then spawn the
#     processor in a detached subprocess while capturing stdout/stderr to worker.log.
#     """
#     job_id = uuid.uuid4().hex[:12]
#     job_dir = make_job_dir(job_id)
#     in_path = job_dir / "input.mp4"
#
#     # save uploaded file
#     with open(in_path, "wb") as f:
#         content = await file.read()
#         f.write(content)
#
#     # initial status
#     write_status(str(job_dir), {'stage': 'queued', 'progress': 0.01})
#
#     # spawn worker with logfile
#     python_bin = shutil.which("python3") or shutil.which("python") or "python"
#     backend_dir = str(ROOT)  # ensure processor can be imported
#
#     # prepare inline Python script to run the pipeline
#     # using raw string literals in the inline script to preserve backslashes on Windows
#     inline_script = (
#         "import sys; "
#         f"sys.path.insert(0, r'{backend_dir}'); "
#         f"from processor import run_pipeline; "
#         f"run_pipeline(r'{str(job_dir)}', r'{str(in_path)}')"
#     )
#
#     log_path = job_dir / "worker.log"
#     # spawn process without a shell (safer quoting). stdout/stderr go to the log file.
#     with open(log_path, "ab") as lf:
#         subprocess.Popen([python_bin, "-u", "-c", inline_script], stdout=lf, stderr=lf)
#
#     return {"job_id": job_id}


@app.get("/status")
def status(job_id: str):
    job_dir = OUTPUTS / job_id
    if not job_dir.exists():
        return JSONResponse({"error": "job not found"}, status_code=404)
    s = read_status(str(job_dir))
    return s


@app.get("/download")
def download(job_id: str):
    job_dir = OUTPUTS / job_id
    f = job_dir / 'vr180_sbs.mp4'
    if not f.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(str(f), media_type='video/mp4', filename='vr180_sbs.mp4')


# small health-check
@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "ok"
