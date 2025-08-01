# MC-Distil: Meta-Collaborative Distillation

**Official PyTorch implementation of "Unified Wisdom: Harnessing Collaborative Learning to Improve Efficacy of Knowledge Distillation"**

## Abstract

We address the ineffectiveness of knowledge distillation when teacher and student models have significant capacity gaps via **"meta-collaborative distillation" (MC-Distil)**, where students of varying capacities collaborate during distillation using a "coordinator" network (C-Net). We achieve **average gains of 2.5% on CIFAR-100 and 2% on Tiny ImageNet** with mere **5% training overhead** and **no extra inference cost**.

## Installation

1. **Clone the repository**:
```bash
git clone git@github.com:AtharvaTambat/MC-Distil.git
cd MC-Distil
```

2. **Install dependencies**:
```bash
# Option 1: Using pip with requirements.txt
pip install -r requirements.txt

# Option 2: Using conda with environment.yml
conda env create -f environment.yml
conda activate mc-distil

# Option 3: Install as a package (recommended)
pip install -e .
```

3. **Verify installation**:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import mc_distil; print('MC-Distil installed successfully!')"
```

## Usage

**Note**: Before running the scripts, open the `.sh` files in `mc_distil/scripts/` and modify the input parameters (dataset path, model names, batch size, etc.) according to your setup.

### Standard Knowledge Distillation (Baseline)
```bash
# 1. Edit parameters in runner_kd.sh as needed
# 2. Run the script
cd mc_distil/scripts
bash runner_kd.sh
```

### MC-Distil (Our Method)
```bash
# 1. Edit parameters in runner_meta.sh as needed  
# 2. Run the script
cd mc_distil/scripts
bash runner_meta.sh
```

## Dataset Preparation

### Supported Datasets
- **CIFAR-10/100**: Automatically downloaded
- **Tiny ImageNet**: Download from [official source](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- **Custom datasets**: Support for standard PyTorch datasets

### Data Directory Structure
```
data/
├── cifar10/          # Auto-downloaded by torchvision
├── cifar100/         # Auto-downloaded by torchvision  
└── tiny-imagenet-200/   # Manual download required
    ├── train/
    ├── val/
    └── test/
```

### Download Tiny ImageNet
```bash
# Create data directory
mkdir -p data
cd data

# Download and extract Tiny ImageNet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
cd ..
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.