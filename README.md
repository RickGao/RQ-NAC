# Residual Quantization with N-gram–driven Arithmetic Coding (RQ-NAC)

RQ-NAC is an efficient and lightweight image compression framework combining **Residual Quantization (RQ)** with **N-gram–driven Arithmetic Coding (NAC)**.  
It provides adaptive bitrate control, low computation overhead and competitive rate–distortion performance for practical image transmission.

---

## Environment

```bash
# Create Conda environment
conda create -n rqnac python=3.10.18
conda activate rqnac

# Install PyTorch
pip install torch torchvision

# Install project dependencies
pip install -r requirements.txt

```


---


## Training and Evaluation of RQ-VAE

```bash
# Go to RQ-VAE folder
cd rq-vae

# Train RQ-VAE
torchrun
# Example

# Evaluate RQ-VAE
python compute_recon.py --split=val --vqvae=$RQVAE_CKPT
# Example
python compute_recon.py --split=val --vqvae=$RQVAE_CKPT

```


---


## Training and Evaluation of NAC

```bash

# Run NAC
python nac.py


```
