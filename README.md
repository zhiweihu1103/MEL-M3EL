# Multi-level Matching Network for Multimodal Entity Linking
#### This repo provides the source code & data of our paper: Multi-level Matching Network for Multimodal Entity Linking(KDD2025).

## Dependencies
* conda create -n m3el python=3.7 -y
* torch==1.11.0+cu113
* transformers==4.27.1
* torchmetrics==0.11.0
* tokenizers==0.12.1
* pytorch-lightning==1.7.7
* omegaconf==2.2.3
* pillow==9.3.0

**Note:** We found that the version of transformers affects performance.

## Running the code
### Dataset
1. Download the datasets from [MIMIC paper](https://github.com/pengfei-luo/MIMIC).
2. Create the root directory ./data and put the dataset in.
3. Download the pretrained_weight from [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).
4. Create the root directory ./checkpoint and put the pretrained_weight in.

### Training model
```python
sh run.sh
```
**Note:** We provide commands for running three datasets in run.sh. You can switch commands by opening comments. 

### Training logs
**Note:** We provide logs of our training in the logs directory.

## Citation
If you find this code useful, please consider citing the following paper.
```
@article{
  author={Zhiwei Hu and Víctor Gutiérrez-Basulto and Ru Li and Jeff Z. Pan},
  title={Multi-level Matching Network for Multimodal Entity Linking},
  publisher="ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
  year={2025}
}
```
## Acknowledgement
We refer to the code of [MIMIC](https://github.com/pengfei-luo/MIMIC. Thanks for their contributions.
