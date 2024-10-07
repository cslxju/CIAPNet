# Category-based Interactive Attention and Perception Fusion Network for Semantic Segmentation of Remote Sensing Images

## Datasets
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

## Install
```
conda create -n ciap python=3.8
conda activate ciap
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```
## Data Preprocessing

Please follw the [GeoSeg](https://github.com/WangLibo1995/GeoSeg) to preprocess the LoveDA, Potsdam and Vaihingen dataset.

## Training
"-c" means the path of the config, use different **config** to train different models.

```shell
python CIAPNet/train_supervision.py -c CIAPNet/config/potsdam/ciapnet.py
```

```shell
python CIAPNet/train_supervision.py -c CIAPNet/config/vaihingen/ciapnet.py
```

```shell
python CIAPNet/train_supervision.py -c CIAPNet/config/loveda/ciapnet.py
```
## Testing
**Vaihingen**
```shell
python CIAPNet/test_vaihingen.py -c CIAPNet/config/vaihingen/ciapnet.py -o ~/fig_results/ciapnet_vaihingen/ --rgb -t "d4"
```

**Potsdam**
```shell
python CIAPNet/test_potsdam.py -c CIAPNet/config/potsdam/ciapnet.py -o ~/fig_results/ciapnet_potsdam/ --rgb -t "d4"
```

**LoveDA** 

```shell
python CIAPNet/test_loveda.py -c CIAPNet/config/loveda/ciapnet.py -o ~/fig_results/ciapnet_loveda --rgb -t "d4"
```

## Acknowledgement

Many thanks the following projects's contributions to **CIAPNet**.
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
