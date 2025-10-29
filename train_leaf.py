import os, glob, torch
from ultralytics import YOLO
from roboflow import Roboflow
from huggingface_hub import HfApi, hf_hub_download, model_info
from huggingface_hub.utils import HfHubHTTPError

# ---------------- NVIDIA optimization switches ----------------
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- ENV ----------------
model_name = os.getenv("MODEL_NAME", "rtdetr-l")
HF_TOKEN   = os.getenv("HF_TOKEN")
HF_REPO    = os.getenv("HF_REPO")
RF_TOKEN   = os.getenv("RF_TOKEN")
if not RF_TOKEN:
    raise RuntimeError("RF_TOKEN is required for Roboflow access.")

S1_FILENAME = f"stage1_{model_name}.pt"  # state_dict save
name_stage1 = "leaf_stage1"
name_stage2 = "leaf_stage2_context_only"

# ---------------- HF setup / check for pretrained ----------------
skip_stage1 = False
api = None
stage1_best = None  # path we'll feed to YOLO() for Stage 2

if HF_TOKEN and HF_REPO:
    api = HfApi(token=HF_TOKEN)
    try:
        info = model_info(HF_REPO, token=HF_TOKEN)

        # check if we've already uploaded best.pt (full YOLO ckpt)
        has_best = any(f.rfilename == "best.pt" for f in info.siblings)
        has_s1   = any(f.rfilename == S1_FILENAME for f in info.siblings)

        if has_best or has_s1:
            print(f"⚡ Found pretrained weights in {HF_REPO}, skipping Stage 1 training.")
            skip_stage1 = True

            # prefer best.pt because Ultralytics can load it directly
            try:
                stage1_best = hf_hub_download(
                    repo_id=HF_REPO,
                    filename="best.pt",
                    token=HF_TOKEN
                )
            except HfHubHTTPError:
                # fallback: stage1_{model}.pt (state_dict only)
                stage1_best = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=S1_FILENAME,
                    token=HF_TOKEN
                )
        else:
            print("ℹ️ Repo found, but no weights yet — will run Stage 1 training.")
    except HfHubHTTPError:
        print("ℹ️ No pretrained model found on Hugging Face — will run Stage 1 training.")
else:
    print("ℹ️ HF_TOKEN/HF_REPO not set: will train fresh and skip HF upload.")

# ---------------- Roboflow dataset/project ----------------
rf = Roboflow(api_key=RF_TOKEN)
workspace = "my-argiculture"
project_name = "leaf_dataset-aejj3"
project = rf.workspace(workspace).project(project_name)

# ---------------- Stage 1 ----------------
if not skip_stage1:
    version = project.version(3)
    dataset = version.download("yolov9")
    STAGE1_DATA_YAML = f"{dataset.location}/data.yaml"

    base_ckpt = f"{model_name}.pt"
    model = YOLO(base_ckpt)

    model.train(
        data=STAGE1_DATA_YAML,
        imgsz=700,
        epochs=80,
        lr0=1e-3,
        optimizer="AdamW",
        cos_lr=True,
        patience=10,
        amp=True,
        batch=-1,             # Auto batch finder
        degrees=180,          # dont change it
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=0.0, copy_paste=0.0, mixup=0.0,
        weight_decay=0.05,
        project="runs",
        name=name_stage1,
        pretrained=True,
        val=True,
        device=device,
    )

    # Ultralytics full checkpoint (this is what YOLO(...) can reload)
    stage1_best = f"runs/{name_stage1}/weights/best.pt"

    # Also save a deterministic state_dict snapshot for HF incremental reuse
    torch.save(
        model.model.state_dict(),
        f"runs/{name_stage1}/weights/{S1_FILENAME}"
    )

# safety check before Stage 2
if not stage1_best:
    raise RuntimeError("Stage 2 requires Stage 1 weights. 'stage1_best' is None.")

# ---------------- Stage 2 dataset prep (context-only filtering) ----------------
version = project.version(5)
dataset = version.download("yolov9")

base = dataset.location
splits = ["train", "valid", "test"]
exts = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

for split in splits:
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")

    # collect all CONTEXTIMG_* basenames (no extension)
    context_names = set()
    for ext in exts:
        for path in glob.glob(f"{img_dir}/CONTEXTIMG_*.{ext}"):
            context_names.add(os.path.splitext(os.path.basename(path))[0])

    deleted_count = 0
    for ext in exts:
        for img_path in glob.glob(f"{img_dir}/*.{ext}"):
            name_noext = os.path.splitext(os.path.basename(img_path))[0]
            if name_noext not in context_names:
                # drop both img and its label if exists
                os.remove(img_path)
                label_path = os.path.join(lbl_dir, f"{name_noext}.txt")
                if os.path.exists(label_path):
                    os.remove(label_path)
                deleted_count += 1

    print(f"✅ [{split}] Kept {len(context_names)} CONTEXTIMG_* files. Deleted {deleted_count} others.")

STAGE2_DATA_YAML = os.path.join(dataset.location, "data.yaml")

freeze  = int(os.getenv("FREEZE_LAYERS", 10))
epochs  = int(os.getenv("EPOCHS", 30))
batch   = int(os.getenv("BATCH", 4))

# ---------------- Stage 2 fine-tune (context only) ----------------
model = YOLO(stage1_best)
model.train(
    data=STAGE2_DATA_YAML,
    imgsz=1280,
    epochs=epochs,
    lr0=2e-5,
    optimizer="AdamW",
    cos_lr=True,
    patience=8,
    amp=True,
    batch=batch,
    freeze=freeze,
    weight_decay=0.005,
    degrees=10,
    translate=0.05,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.2,
    scale=0.10,
    shear=0.0,
    perspective=0.0,
    project="runs",
    name=name_stage2,
    pretrained=True,
    val=True,
    device=device,
)

# ---------------- Export model ----------------
model.export(format="onnx", dynamic=True)

# ---------------- Upload to HF (optional) ----------------
if api:
    if not skip_stage1:
        api.upload_folder(
            folder_path=f"runs/{name_stage1}/weights",
            repo_id=HF_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )
    # Optionally also upload Stage 2 folder (fine-tuned + onnx)
    api.upload_folder(
        folder_path=f"runs/{name_stage2}/weights",
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"✅ Stage 1 + Stage 2 + ONNX uploaded to Hugging Face repo {HF_REPO}")

torch.cuda.empty_cache()
