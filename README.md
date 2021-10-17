# LayoutTransformer

[arXiv](https://arxiv.org/abs/2006.14615) | [BibTeX](#bibtex) | [Project Page](https://kampta.github.io/layout)

This repo contains code for single GPU training of LayoutTransformer from
[LayoutTransformer: Layout Generation and Completion with Self-attention](https://arxiv.org/abs/2006.14615).
This code was rewritten from scratch using a cleaner GPT [codebase](https://github.com/karpathy/minGPT).
Some of the details such as training hyperparameters might differ from the arxiv version of the paper.

<!-- ![teaser!](imgs/layout_teasor.jpg?raw=true) -->


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
```

In your wandb, you can see some generated samples

![media_images_sample_random_layouts_18750_0](https://user-images.githubusercontent.com/1719140/137636972-4030c68e-b1c1-4234-b420-cf3068a5a9c6.png)
![media_images_sample_random_layouts_18750_1](https://user-images.githubusercontent.com/1719140/137636974-0f40c6ce-ea3c-445f-9610-b660f8b60d38.png)
![media_images_sample_random_layouts_18750_2](https://user-images.githubusercontent.com/1719140/137636975-8365f231-246d-4aae-a2a2-a339dd27e8b5.png)
![media_images_sample_random_layouts_18750_3](https://user-images.githubusercontent.com/1719140/137636976-6c8b88c0-41c0-43e1-a492-17dc718138be.png)


```
# Training on COCO bounding boxes or PubLayNet
python main.py \
    --train_json /path/to/annotations/train.json \
    --val_json /path/to/annotations/val.json \
    --exp publaynet
```

For the PubLayNet dataset, generated samples might look like this

<!-- ![media_images_sample_random_layouts_15738_0](https://user-images.githubusercontent.com/1719140/137637044-cc345ae4-49c1-4ae2-ad2e-5532d2a080f6.png) -->
![media_images_sample_random_layouts_15738_3](https://user-images.githubusercontent.com/1719140/137637046-e2181cda-904e-4ea3-868b-39a7bf64a236.png)
![media_images_sample_random_layouts_26230_2](https://user-images.githubusercontent.com/1719140/137637047-43fd285f-afec-42ba-a4f7-04ddf66d4d86.png)
![media_images_sample_random_layouts_26230_3](https://user-images.githubusercontent.com/1719140/137637048-7263f9ab-1d19-4826-a6c2-d7ce152d9e0d.png)


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
* https://github.com/ChrisWu1997/PQ-NET


## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
