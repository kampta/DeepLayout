# LayoutTransformer

[arXiv](https://arxiv.org/abs/2006.14615) | [BibTeX](#bibtex) | [Project Page](https://kampta.github.io/layout)

This repo contains code for single GPU training of LayoutTransformer from
[LayoutTransformer: Layout Generation and Completion with Self-attention](https://arxiv.org/abs/2006.14615).
This code was rewritten from scratch using a cleaner GPT [codebase](https://github.com/karpathy/minGPT).
Some of the details such as training hyperparameters might differ from the arxiv version of the paper.

![teaser!](imgs/layout_teasor.jpg?raw=true)


## How To Use This Code

Start a new conda environment
```
conda env create -f environment.yml
conda activate layout
```
or update an existing environment

```
conda env update -f environment.yml --prune
```

### Logging with `wandb`

In order to log experiments to wandb, 
we use wandb's API keys that can be found here https://wandb.ai/settings.
Copy your key and store them in an environment variable using

```
export WANDB_API_KEY=<Your WANDB API KEY>
```

Alternately, you can also login using `wandb login`.

## Datasets

### COCO Bounding Boxes

See the instructions to obtain the dataset [here](https://cocodataset.org/).

### PubLayNet Document Layouts

See the instructions to obtain the dataset [here](https://github.com/ibm-aur-nlp/PubLayNet). 


## LayoutVAE

Reimplementation of [LayoutVAE](https://arxiv.org/abs/1907.10719) is [here](layout_vae).
Code contributed primarily by Justin.

```
cd layout_vae

# Train the CountVAE model
python train_counts.py \
    --exp count_coco_instances \
    --train_json /path/to/coco/annotations/instances_train2017.json \
    --val_json /path/to/coco/annotations/instances_val2017.json \
    --epochs 50

# Train the BoxVAE model
python train_counts.py \
    --exp box_coco_instances \
    --train_json /path/to/coco/annotations/instances_train2017.json \
    --val_json /path/to/coco/annotations/instances_val2017.json \
    --epochs 50
```

## LayoutTransformer

Rewritten from scratch using a cleaner GPT [codebase](https://github.com/karpathy/minGPT).
Some of the details such as training hyperparameters might differ from the arxiv version.

```
# Training on MNIST layouts
python main.py \
    --data_dir /path/to/mnist \
    --threshold 1 --exp mnist_threshold_1
    
# Training on COCO bounding boxes or PubLayNet
python main.py \
    --train_json /path/to/annotations/train.json \
    --val_json /path/to/annotations/val.json \
    --exp publaynet
```

## BibTeX

If you use this code, please cite
```text
@inproceedings{gupta2021layouttransformer,
  title={LayoutTransformer: Layout Generation and Completion with Self-attention},
  author={Gupta, Kamal and Lazarow, Justin and Achille, Alessandro and Davis, Larry S and Mahadevan, Vijay and Shrivastava, Abhinav},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1004--1014},
  year={2021}
}
}
```

## Acknowledgments

We would like to thank several public repos

* https://github.com/JiananLi2016/LayoutGAN-Tensorflow
* https://github.com/Layout-Generation/layout-generation
* https://github.com/karpathy/minGPT


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
