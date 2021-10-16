# Cartoon-StyleGAN 🙃 : Fine-tuning StyleGAN2 for Cartoon Face Generation

> **Abstract**
>
> Recent studies have shown remarkable success in the unsupervised image to image (I2I) translation. However, due to the imbalance in the data, learning joint distribution for various domains is still very challenging. Although existing models can generate realistic target images, it’s difficult to maintain the structure of the source image. In addition, training a generative model on large data in multiple domains requires a lot of time and computer resources. To address these limitations, I propose a novel image-to-image translation method that generates images of the target domain by finetuning a stylegan2 pretrained model. The stylegan2 model is suitable for unsupervised I2I translation on unbalanced datasets; it is highly stable, produces realistic images, and even learns properly from limited data when applied with simple fine-tuning techniques. Thus, in this project, I propose new methods to preserve the structure of the source images and generate realistic images in the target domain.

<p align='center'><img src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/Result.gif?raw=1" width = '700'></p>

**Inference Notebook**

🎉 You can do this task in colab ! : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/happy-jihye/Cartoon-StyleGan2/blob/main/Cartoon_StyleGAN2.ipynb)

**Arxiv**
[![arXiv](https://img.shields.io/badge/arXiv-2010.05334-b31b1b.svg)](https://arxiv.org/abs/2106.12445)

**[NEW!] 2021.08.30 Streamlit Ver**
- [`cartoon-stylegan streamlit inference repo`](https://github.com/happy-jihye/Streamlit-Tutorial/tree/main/cartoon-stylegan)
<p align='center'><img src="https://github.com/happy-jihye/Streamlit-Tutorial/blob/main/asset/cartoon-stylegan-1.gif?raw=1?raw=1" width = '700'></p>


---

## 1. Method

### Baseline : StyleGAN2-ADA + FreezeD

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/baseline.PNG?raw=1' width = '700' ></p>

It generates realistic images, but does not maintain the structure of the source domain.


|      |      |
| ---- | ---- |
| <img src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/baseline-rom101.gif?raw=1">  |     <img  src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/baseline-simpson.gif?raw=1">   |


### Ours : FreezeSG (Freeze Style vector and Generator)

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/latent.PNG?raw=1' width = '800' ></p>

[FreezeG](https://github.com/bryandlee/FreezeG) is effective in maintaining the structure of the source image. As a result of various experiments, I found that not only the initial layer of the generator but also the initial layer of the style vector are important for maintaining the structure. Thus, I froze the low-resolution layer of both the generator and the style vector.


**Freeze Style vector and Generator**
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/FreezeSG.PNG?raw=1' width = '800' ></p>

**Results**
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/FreezeSG2.PNG?raw=1' width = '800' ></p>

**With [Layer Swapping](https://arxiv.org/abs/2010.05334)**

When LS is applied, the generated images by FreezeSG have a higher similarity to the source image than when FreezeG or the baseline (FreezeD + ADA) were used. However, since this fixes the weights of the low-resolution layer of the generator, it is difficult to obtain meaningful results when layer swapping on the low-resolution layer.
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/FreezeSG3.png?raw=1' width = '800' ></p>

### Ours : Structure Loss

Based on the fact that the structure of the image is determined at low resolution, I apply structure loss to the values of the low-resolution layer so that the generated image is similar to the image in the source domain. The structure loss makes the RGB output of the source generator to be fine-tuned to have a similar value with the RGB output of the target generator during training.

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/StructureLoss.png?raw=1' width = '700' ></p>
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/StructureLoss2.PNG?raw=1' width = '700' ></p>

**Results**

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/StructureLoss3.PNG?raw=1' width = '700' ></p>
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/StructureLoss4.png?raw=1' width = '700' ></p>

### Compare
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/compare.PNG?raw=1' width = '700' ></p>

---
## 2. Application : Change Facial Expression / Pose

I applied various models(ex. Indomain-GAN, SeFa, StyleCLIP…) to change facial expression, posture, style, etc.

### (1) Closed Form Factorization(SeFa)

- [Closed-Form Factorization of Latent Semantics in GANs](https://arxiv.org/abs/2007.06600)

**Pose**
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/sefa-pose.gif?raw=1' width = '700' ></p>

**Slim Face**
<p align='center'><img src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/sefa.gif?raw=1" width = '700'></p>

### (2) StyleCLIP – Latent Optimization

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/styleclip.PNG?raw=1' width = '700' ></p>

Inspired by [StyleCLIP](https://arxiv.org/abs/2103.17249) that manipulates generated images with text, I change the faces of generated
cartoon characters by text. I used the latent optimization method among the three methods of StyleCLIP and additionally introduced styleclip strength. It allows the latent vector to linearly move in the direction of the optimized latent vector, making the image change better with text.

**with baseline model(FreezeD)**
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/styleclip2.PNG?raw=1' width = '700' ></p>
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/styleclip5.PNG?raw=1' width = '700' ></p>

**with our model(structureLoss)**
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/styleclip3.PNG?raw=1' width = '700' ></p>
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/styleclip4.PNG?raw=1' width = '700' ></p>


### (3) Style Mixing

Style-Mixing
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylemixing5.png?raw=1' width = '700' ></p>

When mixing layers, I found specifics layers that make a face. While the overall structure (hair style, facial shape, etc.) and texture (skin color and texture) were maintained, only the face(eyes, nose and mouth) was changed.

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylemixing.PNG?raw=1' width = '700' ></p>
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylemixing2.PNG?raw=1' width = '700' ></p>

**Results**

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylemixing3.PNG?raw=1' width = '700' ></p>
<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylemixing4.PNG?raw=1' width = '700' ></p>


---

## 3. Requirements

I have tested on:

- PyTorch 1.8.0, CUDA 11.1
- Docker : [`pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel`](https://hub.docker.com/r/pytorch/pytorch)


### Installation

Clone this repo :

```bash
git clone https://github.com/happy-jihye/Cartoon-StyleGan2
cd Cartoon-StyleGan2
```

### Pretrained Models

Please download the pre-trained models from the following links. 


| Path | Description
| :--- | :----------
|[StyleGAN2-FFHQ256](https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view)  | StyleGAN2 pretrained model(256px) with FFHQ dataset from [Rosinality](https://github.com/rosinality/stylegan2-pytorch)
|[StyleGAN2-Encoder](https://drive.google.com/uc?id=1QQuZGtHgD24Dn5E21Z2Ik25EPng58MoU)  | In-Domain GAN Inversion model with FFHQ dataset from [Bryandlee](https://github.com/bryandlee/stylegan2-encoder-pytorch)|
|[NaverWebtoon](https://drive.google.com/uc?id=1yIn_gM3Fk3RrRphTPNBPgJ3c-PuzCjOB)  | FreezeD + ADA with [NaverWebtoon Dataset](https://www.webtoons.com/en/) |
|[NaverWebtoon_FreezeSG](https://drive.google.com/uc?id=1OysFtj7QTy7rPnxV9TXeEgBfmtgr8575)  | FreezeSG with [NaverWebtoon Dataset](https://www.webtoons.com/en/) |
|[NaverWebtoon_StructureLoss](https://drive.google.com/uc?id=1Oylfl5j-XGoG_pFHtQwHd2G7yNSHx2Rm)  | StructureLoss with [NaverWebtoon Dataset](https://www.webtoons.com/en/) |
|[Romance101](https://drive.google.com/uc?id=1wWt4dPC9TJfJ6cF3mwg7kQvpuVwPhSN7)  | FreezeD + ADA with [Romance101 Dataset](https://www.webtoons.com/en/romance/romance-101/list?title_no=2406&page=1) |
|[TrueBeauty](https://drive.google.com/uc?id=1yEky49SnkBqPhdWvSAwgK5Sbrc3ctz1y)  | FreezeD + ADA with [TrueBeauty Dataset](https://www.webtoons.com/en/romance/truebeauty/list?title_no=1436&page=1) |
|[Disney](https://drive.google.com/uc?id=1z51gxECweWXqSYQxZJaHOJ4TtjUDGLxA)  | FreezeD + ADA with [Disney Dataset](https://github.com/justinpinkney/toonify) |
|[Disney_FreezeSG](https://drive.google.com/uc?id=1PJaNozfJYyQ1ChfZiU2RwJqGlOurgKl7)  | FreezeSG with [Disney Dataset](https://github.com/justinpinkney/toonify) |
|[Disney_StructureLoss](https://drive.google.com/uc?id=1PILW-H4Q0W8S22TO4auln1Wgz8cyroH6)  | StructureLoss with [Disney Dataset](https://github.com/justinpinkney/toonify) |
|[Metface_FreezeSG](https://drive.google.com/uc?id=1P5T6DL3Cl8T74HqYE0rCBQxcq15cipuw)  | FreezeSG with [Metface Dataset](https://github.com/NVlabs/metfaces-dataset) |
|[Metface_StructureLoss](https://drive.google.com/uc?id=1P65UldIHd2QfBu88dYdo1SbGjcDaq1YL)  | StructureLoss with [Metface Dataset](https://github.com/NVlabs/metfaces-dataset) |

If you want to download all of the pretrained model, you can use `download_pretrained_model()` function in `utils.py`.

**Dataset**

I experimented with a variety of datasets, including Naver Webtoon, Metfaces, and Disney. 

[NaverWebtoon Dataset](https://www.webtoons.com/en/) contains facial images of webtoon characters serialized on Naver. I made this dataset by [crawling webtoons from Naver’s webtoons site](https://happy-jihye.github.io/notebook/python-3/) and [cropping the faces](https://github.com/nagadomi/lbpcascade_animeface) to 256 x 256 sizes. There are about 15 kinds of webtoons and 8,000 images(not aligned). I trained the entire Naver Webtoon dataset, and I also trained each webtoon in this experiment

<p align='center'><img src='https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/naverwebtoon_dataset.PNG?raw=1' width = '600' ></p>

I was also allowed to share a pretrained model with writers permission to use datasets. Thank you for the writers ([Yaongyi](https://www.webtoons.com/en/romance/truebeauty/list?title_no=1436&page=1), [Namsoo](https://www.webtoons.com/en/romance/romance-101/list?title_no=2406&page=1), [justinpinkney](https://github.com/justinpinkney/toonify)) who gave us permission.

## Getting Started !

**1. Prepare LMDB Dataset**

First create lmdb datasets:

```
python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

# if you have zip file, change it to lmdb datasets by this commend
python run.py --prepare_data=DATASET_PATH --zip=ZIP_NAME --size SIZE
```


**2. Train**

```
# StyleGAN2
python train.py --batch BATCH_SIZE LMDB_PATH
# ex) python train.py --batch=8 --ckpt=ffhq256.pt --freezeG=4 --freezeD=3 --augment --path=LMDB_PATH

# StructureLoss
# ex) python train.py --batch=8 --ckpt=ffhq256.pt --structure_loss=2 --freezeD=3 --augment --path=LMDB_PATH

# FreezeSG
# ex) python train.py --batch=8 --ckpt=ffhq256.pt --freezeStyle=2 --freezeG=4 --freezeD=3 --augment --path=LMDB_PATH


# Distributed Settings
python train.py --batch BATCH_SIZE --path LMDB_PATH \
    -m torch.distributed.launch --nproc_per_node=N_GPU --main_port=PORT
```

**Options**

1. Project images to latent spaces

    ```
    python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...
    ```

2. [Closed-Form Factorization](https://arxiv.org/abs/2007.06600)

    You can use `closed_form_factorization.py` and `apply_factor.py` to discover meaningful latent semantic factor or directions in unsupervised manner.

    First, you need to extract eigenvectors of weight matrices using `closed_form_factorization.py`

    ```
    python closed_form_factorization.py [CHECKPOINT]
    ```

    This will create factor file that contains eigenvectors. (Default: factor.pt) And you can use `apply_factor.py` to test the meaning of extracted directions

    ```
    python apply_factor.py -i [INDEX_OF_EIGENVECTOR] -d [DEGREE_OF_MOVE] -n [NUMBER_OF_SAMPLES] --ckpt [CHECKPOINT] [FACTOR_FILE]
    # ex) python apply_factor.py -i 19 -d 5 -n 10 --ckpt [CHECKPOINT] factor.pt
    ```

---

## StyleGAN2-ada + FreezeD


During the experiment, I also carried out a task to generate a cartoon image based on [Nvidia Team's StyleGAN2-ada code](https://github.com/NVlabs/stylegan2-ada-pytorch). When training these models, I didn't control the dataset resolution(256px)😂. So the quality of the generated image can be broken. 

You can practice based on this code at Colab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/happy-jihye/Cartoon-StyleGan2/blob/main/stylegan2_ada_freezeD.ipynb)

|  Generated-Image   |  Interpolation    |
| ---- | ---- |
| <img src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylegan2ada-sim-image.png?raw=1">  |     <img  src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylegan2ada-sim-interpolation.gif?raw=1">   |
| <img src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylegan2ada-love-multi.png?raw=1">  |     <img  src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylegan2ada-love-interpolation.gif?raw=1">   |
| <img src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylegan2ada-rom-multi.png?raw=1">  |     <img  src="https://github.com/happy-jihye/Cartoon-StyleGan2/blob/main/asset/images/stylegan2ada-rom-interpolation.gif?raw=1">   |

## Reference

- [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
- [bryandlee/FreezeG](https://github.com/bryandlee/FreezeG)
- [justinpinkney/toonify](https://github.com/justinpinkney/toonify)
