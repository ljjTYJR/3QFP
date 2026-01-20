# 3QFP
<p align="center">
  	<h2 align="center">
      3QFP: Efficient neural implicit surface reconstruction using Tri-Quadtrees and Fourier feature Positional encoding [ICRA24]
	</h2>
	<h2 align="center">
        <a href="https://arxiv.org/abs/2401.07164">Arxiv</a>
	</h2>
</p>

# Overview

**Overview of our method.**
<p align="center">
    <a href="">
    	<img src="./assets/teaser.jpg" alt="teaser" width="100%">
    </a>
    <a href="">
        <img src="./assets/caption.png" alt="caption" width="100%">
    </a>
</p>

<!-- We represent the scene with three planar quadtrees $\mathcal{M}_{i}^{\ell}$, $i \in \{XZ,YZ,XY\}$ and $\ell$ represents the quadtree depth. We store features in the deepest $H$ levels of resolution of quadtrees. When querying for a point $\mathbf{p}$, we project it onto planar quadtrees to identify the node containing $\mathbf{p}$ at the level $\ell$. The feature of $\mathbf{p}$ is then calculated by bilinear interpolation based on the queried location and vertex features. We add features at the same level and concatenate among different levels. Concatenated with the positional encoding $\gamma(\mathbf{p})$, $\mathbf{p}$'s feature~($\Phi(\mathbf{p})$) is fed into a small MLP~($\mathcal{F}_\Theta$) to predict the SDF value. The learnable features stored in the quadtree nodes and the network parameters are optimized in real-time using the loss function $\mathcal{L}_{\text{bce}}$. The learnable feature vectors have length $d$ and the positional encoding feature vector has length $6m$. -->

# Installation

The code is based on the implementation of [SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping).

## Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies** (creates virtual environment with Python 3.9):
```bash
uv pip install -e . --index-strategy unsafe-best-match
```

This will automatically install:
- PyTorch 1.12.1+cu116 (CUDA 11.6, compatible with CUDA 11.6/11.7/11.8)
- torchvision 0.13.1+cu116
- torchaudio 0.12.1+cu116
- kaolin 0.12.0 (pre-built for PyTorch 1.12.1)
- torch-scatter 2.1.0+cu116
- tinycudann (built from source)
- All other dependencies

**Note:** `--index-strategy unsafe-best-match` is required to use both PyTorch and Kaolin custom indexes.

3. **Verify installation**:
```bash
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"
.venv/bin/python -c "import tinycudann; print('tinycudann: OK')"
.venv/bin/python -c "import kaolin as kal; print(f'kaolin: {kal.__version__}')"
```

**Note:** The project uses PyTorch 1.12.1+cu116 with pre-built kaolin 0.12.0 for compatibility. CUDA 11.6 binaries work with CUDA 11.6/11.7/11.8 hardware.

## Method 2: Using conda/pip

1. **Create a conda environment**:
```bash
conda create --name 3qfp python=3.9
conda activate 3qfp
```

2. **Install torch-related packages (CUDA 11.6)**:
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. **Install kaolin and torch-scatter**:
```bash
pip install kaolin==0.12.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu116.html
pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
```

4. **Install other dependencies**:
```bash
pip install numpy tqdm wandb open3d scikit-image natsort pyquaternion pyyaml scipy
pip install "werkzeug>=2.0,<3.0" "flask>=2.0,<3.0"
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn.git@master#subdirectory=bindings/torch
```

# Dataset
Also, similarly, we suggest the download scripts from SHINE-Mapping.
- `MaiCity` dataset
```
sh ./scripts/download_maicity.sh
```
- `KITTI` dataset
```
sh ./scripts/download_kitti_example.sh
```
- `Newer College`
```
sh ./scripts/download_ncd_example.sh
```
In the configuration (.yaml) files, you can specify the dataset path.

`pc_path`: the folder containing the point cloud (.bin, .ply or .pcd format) for each frame.
`pose_path` : the pose file (.txt) containing the transformation matrix of each frame.
`calib_path` : the calib file (.txt) containing the static transformation between sensor and body frames (optional, would be identity matrix if set as '').

# Run

## Command Line
```bash
python run.py ./config/maicity/maicity_batch.yaml
```

Or with uv:
```bash
.venv/bin/python run.py ./config/radar.yaml
```