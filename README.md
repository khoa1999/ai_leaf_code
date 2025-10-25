# AI Leaf Code

AI Leaf Code is a compact YOLO-based pipeline for detecting diseases on plant leaves. The repository includes a configurable training script, lightweight dependencies, and Docker support so you can experiment quickly—whether you prefer local Python environments or containerized workflows.

## Project Overview
- **Model training**: [`train_leaf.py`](train_leaf.py) orchestrates YOLO training for your custom leaf dataset.
- **Dependencies**: Managed through [`requirements.txt`](requirements.txt) for reproducible Python environments.
- **Container image**: The provided [`Dockerfile`](Dockerfile) encapsulates the training environment so experiments can run identically anywhere.

## Quick Start (Local Python)
1. Create and activate a virtual environment.
2. Install dependencies.
3. Launch training with your dataset configuration.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python train_leaf.py --data path/to/dataset.yaml --epochs 100
```

## Git Workflow Cheatsheet
Keep your experiments organized with Git:

```bash
# Clone the repository and enter it
git clone https://github.com/your-username/ai_leaf_code.git
cd ai_leaf_code

# Create a feature branch for your work
git checkout -b feature/improve-dataset

# Stage and commit your changes
git status
git add train_leaf.py data/
git commit -m "Improve dataset preprocessing"

# Push your branch and open a pull request
git push -u origin feature/improve-dataset
```

## Docker Workflow
Run the training pipeline inside a container for full reproducibility.

```bash
# Build the image (tags it as ai-leaf:latest)
docker build -t ai-leaf:latest .

# Run training inside the container, mounting your dataset
mkdir -p ~/leaf-data
# Copy your dataset files into ~/leaf-data before running

docker run --rm \
  -v ~/leaf-data:/workspace/data \
  -e RF_TOKEN=your_roboflow_token \
  -e DATA_CONFIG=/workspace/data/dataset.yaml \
  ai-leaf:latest \
  python train_leaf.py --data "$DATA_CONFIG"
```

### Environment variables
The container respects the following environment variables. Set them with `-e VAR=value` when running `docker run`.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `RF_TOKEN` | ✅ | _none_ | Roboflow API token needed to download the training dataset. |
| `DATA_CONFIG` | ✅ (for the example above) | _none_ | Path (inside the container) to the dataset YAML passed to `train_leaf.py`. |
| `MODEL_NAME` | ❌ | `rtdetr-l` | Base checkpoint to use for Stage 1 pretraining. |
| `EPOCHS` | ❌ | `30` | Stage 2 fine-tuning epochs used inside the script. |
| `BATCH` | ❌ | `4` | Stage 2 batch size. |
| `FREEZE_LAYERS` | ❌ | `10` | Number of layers to freeze during Stage 2 fine-tuning. |
| `HF_TOKEN` | ❌ | _none_ | Hugging Face token for downloading/uploading checkpoints. |
| `HF_REPO` | ❌ | _none_ | Hugging Face repository used in conjunction with `HF_TOKEN`. |

> ℹ️ Environment variables marked “Required” are the minimum needed for the Docker command to succeed. Optional variables unlock full fine-tuning and model publishing capabilities.

### Full fine-tuning example
To override every tunable setting, set the relevant environment variables before invoking the container:

```bash
docker run --rm \
  -v ~/leaf-data:/workspace/data \
  -e RF_TOKEN=your_roboflow_token \
  -e DATA_CONFIG=/workspace/data/dataset.yaml \
  -e MODEL_NAME=yolov9c \
  -e EPOCHS=60 \
  -e BATCH=8 \
  -e FREEZE_LAYERS=0 \
  -e HF_TOKEN=your_hf_token \
  -e HF_REPO=your-username/leaf-detector \
  ai-leaf:latest \
  python train_leaf.py --data "$DATA_CONFIG"
```

With `HF_TOKEN` and `HF_REPO` defined, the script automatically uploads the generated weights back to Hugging Face after training completes.

## Tips for Better Results
- Start with a small number of epochs to validate your setup, then scale up.
- Monitor GPU utilization if available to tune batch sizes and image sizes.
- Version your datasets and label schemas so experiments remain comparable.

## Contributing
Issues, ideas, and pull requests are welcome! Follow the Git workflow above, and remember to document noteworthy changes in your commit messages and pull requests.

Happy leaf-spotting! 🌿
