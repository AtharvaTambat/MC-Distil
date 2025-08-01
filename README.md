# MC-Distil: Meta-Collaborative Distillation

**Official PyTorch implementation of "Meta-Collaborative Distillation for Knowledge Distillation"**

## Abstract

We address the ineffectiveness of knowledge distillation when teacher and student models have significant capacity gaps via **"meta-collaborative distillation" (MC-Distil)**, where students of varying capacities collaborate during distillation using a "coordinator" network (C-Net). We achieve **average gains of 2.5% on CIFAR-100 and 2% on Tiny ImageNet** with only **5% training overhead** and **no inference cost**.

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
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

### Standard Knowledge Distillation (Baseline)
```bash
cd mc_distil/scripts
bash runner_kd.sh
```

### MC-Distil (Our Method)
```bash
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


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{mc_distil2024,
  title={Meta-Collaborative Distillation for Knowledge Distillation},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.