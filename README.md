# README

This repo is the official implementation of [**_Graph Inductive Biases in Transformers without Message Passing_**](https://arxiv.org/abs/2305.17589) (Ma et al., ICML 2023)

> The implementation is based on [GraphGPS (Rampasek et al., 2022)](https://github.com/rampasek/GraphGPS).

Python environment setup with Conda

```bash
conda create -n grit python=3.9
conda activate grit 

# please change the cuda/device version as you need

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --trusted-host download.pytorch.org
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
## conda install openbabel fsspec rdkit -c conda-forge
pip install rdkit

pip install torchmetrics==0.9.1
pip install ogb
pip install tensorboardX
pip install yacs
pip install opt_einsum
pip install graphgym 
pip install pytorch-lightning # required by graphgym 
pip install setuptools==59.5.0
# distuitls has conflicts with pytorch with latest version of setuptools

pip install yacs opt_einsum graphgym pytorch-lightning setuptools==59.5.0


# ---- experiment management tools --------
# pip install wandb  # the wandb is used in GraphGPS but not used in GRIT (ours); please verify the usability before using.
# pip install mlflow 
### mlflow server --backend-store-uri mlruns --port 5000

```

### Running GRIT

```bash
# Run
python main.py --cfg configs/GRIT/zinc-GRIT.yaml  wandb.use False accelerator "cuda:0" optim.max_epoch 2000 seed 41 dataset.dir 'xx/xx/data'

# replace 'cuda:0' with the device to use
# replace 'xx/xx/data' with your data-dir (by default './datasets")
# replace 'configs/GRIT/zinc-GRIT.yaml' with any experiments to run
```

### Configurations and Scripts

- Configurations are available under `./configs/GRIT/xxxxx.yaml`
- Scripts to execute are available under `./scripts/xxx.sh`
  - will run 4 trials of experiments parallelly on `GPU:0,1,2,3`.

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{ma2023GraphInductiveBiases,
	title = {Graph {Inductive} {Biases} in {Transformers} without {Message} {Passing}},
	booktitle = {Proc. {Int}. {Conf}. {Mach}. {Learn}.},
	author = {Ma, Liheng and Lin, Chen and Lim, Derek and Romero-Soriano, Adriana and K. Dokania and Coates, Mark and H.S. Torr, Philip and Lim, Ser-Nam},
	year = {2023},
}
```
