# Residual Quantization with N-gram–driven Arithmetic Coding (RQ-NAC)

RQ-NAC is an efficient and lightweight image compression framework combining **Residual Quantization (RQ)** with **N-gram–driven Arithmetic Coding (NAC)**.  
It provides adaptive bitrate control, low computation overhead, and competitive rate–distortion performance for practical image transmission.

---

## Installation & Setup

```bash
# 1. Create Conda environment
conda create -n rqnac python=3.10.18
conda activate rqnac

# 2. Install PyTorch
pip install torch torchvision
# (Optional) Install specific CUDA version
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt
