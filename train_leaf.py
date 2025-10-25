import os, glob, torch
from ultralytics import YOLO
from roboflow import Roboflow
from huggingface_hub import HfApi

# ---------------- NVIDIA optimization switches ----------------
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Roboflow dataset (Stage 1) ----------------
name = "leaf_stage1"
api_key = os.getenv("RF_TOKEN")
rf = Roboflow(api_key=api_key)

workspace = "my-argiculture"
project_name = "leaf_dataset-aejj3"
project = rf.workspace(workspace).project(project_name)
version = project.version(3)
dataset = version.download("yolov9")

STAGE1_DATA_YAML = f"{dataset.location}/data.yaml"

# ---------------- Train RT-DETR (Stage 1) ----------------
model_name = os.getenv("MODEL_NAME", "rtdetr-l")
model = YOLO(f"{model_name}.pt")  # fixed variable reference

model.train(
    data=STAGE1_DATA_YAML,
    imgsz=700,
    epochs=80,
    lr0=1e-3,
    optimizer="AdamW",
    cos_lr=True,
    patience=10,
    amp=True,
    batch=-1,
    degrees=180,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    mosaic=0.0, copy_paste=0.0, mixup=0.0,
    weight_decay=0.05,
    project="runs",
    name=name,
    pretrained=True,
    val=True
)

PATH = f"runs/{name}/weights/stage1_{model_name}.pt"
torch.save(model.model.state_dict(), PATH)  # must use model.model for YOLOv9 objects

# ---------------- Train RT-DETR (Stage 2) ----------------
version = project.version(4)
dataset = version.download("yolov9")

base = dataset.location
splits = ["train", "valid", "test"]
exts = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

for split in splits:
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")

    context_names = set()
    for ext in exts:
        for path in glob.glob(f"{img_dir}/CONTEXTIMG_*.{ext}"):
            context_names.add(os.path.splitext(os.path.basename(path))[0])

    deleted_count = 0
    for ext in exts:
        for img_path in glob.glob(f"{img_dir}/*.{ext}"):
            name = os.path.splitext(os.path.basename(img_path))[0]
            if name not in context_names:
                os.remove(img_path)
                label_path = os.path.join(lbl_dir, f"{name}.txt")
                if os.path.exists(label_path):
                    os.remove(label_path)
                deleted_count += 1

    print(f"✅ [{split}] Kept {len(context_names)} CONTEXTIMG_* files. Deleted {deleted_count} others.")

stage1_best = os.path.join(os.getcwd(), f"runs/{name}/weights/best.pt")
STAGE2_DATA_YAML = os.path.join(dataset.location, "data.yaml")

# Fixed env variable parsing
freeze = int(os.getenv("FREEZE_LAYERS", 10))
epochs = int(os.getenv("EPOCHS", 30))
batch = int(os.getenv("BATCH", 4))

model = YOLO(stage1_best)
model.train(
    data=STAGE2_DATA_YAML,
    imgsz=1280,
    epochs=epochs,
    lr0=5e-5,
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
    blur=0.1,
    project="runs",
    name="leaf_stage2_context_only",
    pretrained=True,
    val=True
)

# ---------------- Export model ----------------
model.export(format="onnx", dynamic=True)

# ---------------- Upload to Hugging Face ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO")  # fixed os.env → os.getenv

api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="runs/leaf_stage1/weights",
    repo_id=HF_REPO,
    repo_type="model",
    token=HF_TOKEN,
)
print(f"✅ Model uploaded to https://huggingface.co/{HF_REPO}")

torch.cuda.empty_cache()
