import os, torch
from ultralytics import YOLO
from roboflow import Roboflow
from huggingface_hub import HfApi
import os, glob

# ---- NVIDIA optimization switches ----
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Roboflow dataset ----
name =  "leaf_stage1"
api_key = os.getenv("RF_TOKEN")
rf = Roboflow(api_key)
workspace =  "my-argiculture"
project_name =  "leaf_dataset-aejj3"
project = rf.workspace(workspace).project(project_name)
version = project.version(3)
dataset = version.download("yolov9")
STAGE1_DATA_YAML = f"{dataset.location}/data.yaml"

# ---- Train RT-DETR (Stage 1) ----
model = YOLO("rtdetr-l.pt")
model.train(
  data=STAGE1_DATA_YAML,
  imgsz=700,
  epochs=80,            # early-stop will likely end 60–70
  lr0=1e-3,             # AdamW baseline
  optimizer="AdamW",
  cos_lr=True,
  patience=10,
  amp=True,
  batch=-1,             # auto (use largest that fits)
  degrees=180,          # random rotation 0..180
  flipud=0.5, fliplr=0.5,
  hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
  mosaic=0.0, copy_paste=0.0, mixup=0.0,   # keep global augs off here
  weight_decay=0.05,
  project="runs", name=name,
  pretrained=True, val=True
)
PATH = f"runs/{name}/weights/stage1.pt"
torch.save(model.state_dict(), PATH)
# ---- Train RT-DETR (Stage 2) ----
version = project.version(4)
dataset = version.download("yolov9")

base = dataset.location
splits = ["train", "valid", "test"]
exts = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

for split in splits:
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")

    # Find all CONTEXTIMG_* basenames
    context_names = set()
    for ext in exts:
        for path in glob.glob(f"{img_dir}/CONTEXTIMG_*.{ext}"):
            context_names.add(os.path.splitext(os.path.basename(path))[0])

    # Delete non-context images + their labels
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

stage1_best = os.path.join(os.getcwd(),f"runs/{name}/weights/best.pt")
STAGE2_DATA_YAML = os.path.join(dataset.location,"data.yaml")
FREEZE_LAYERS = 10
model = YOLO(stage1_best)
model.train(
  data=STAGE2_DATA_YAML,   # <-- TRAIN = context-only
  imgsz=1280,
  epochs=30,
  lr0=5e-5,                # 0.05× of Stage 1
  optimizer="AdamW",
  cos_lr=True,
  patience=8,
  amp=True,
  batch=4,
  weight_decay=0.005,      #allow remmbering,
  # scene-typical, light global augs:
  # mosaic=0.15, copy_paste=0.4, mixup=0.05,
  degrees=10, translate=0.05, fliplr=0.5, flipud=0.0,
  hsv_h=0.01, hsv_s=0.3, hsv_v=0.2, scale=0.10, shear=0.0, perspective=0.0,
  blur=0.1,
  freeze=FREEZE_LAYERS,
  project="runs", name="leaf_stage2_context_only",
  pretrained=True, val=True,
)

# ---- Export model ----
model.export(format="onnx", dynamic=True)
best_weights = f"runs/{name}/weights/best.pt"

# ---- Upload to Hugging Face ----
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = "dangkhoa1999/Tomato_leaf_fruit"

api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="runs/leaf_stage1/weights",
    repo_id=HF_REPO,
    repo_type="model",
    token=HF_TOKEN,
)
print(f"✅ Model uploaded to https://huggingface.co/{HF_REPO}")

torch.cuda.empty_cache()
