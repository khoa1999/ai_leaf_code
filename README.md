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
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
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
  -e DATA_CONFIG=/workspace/data/dataset.yaml \
  ai-leaf:latest \
  python train_leaf.py --data "$DATA_CONFIG" --epochs 100
```

## Tips for Better Results
- Start with a small number of epochs to validate your setup, then scale up.
- Monitor GPU utilization if available to tune batch sizes and image sizes.
- Version your datasets and label schemas so experiments remain comparable.

## Contributing
Issues, ideas, and pull requests are welcome! Follow the Git workflow above, and remember to document noteworthy changes in your commit messages and pull requests.

Happy leaf-spotting! 🌿
