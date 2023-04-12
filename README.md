# Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer

[![arXiv](https://img.shields.io/badge/arXiv-2303.08622-b31b1b.svg)](https://arxiv.org/abs/2303.08622)

## Abstract
> Diffusion models have shown great promise in text-guided image style transfer, but there is a trade-off between style transformation and content preservation due to their stochastic nature. Existing methods require computationally expensive fine-tuning of diffusion models or additional neural network. To address this, here we propose a zero-shot contrastive loss for diffusion models that doesn't require additional fine-tuning or auxiliary networks. By leveraging patch-wise contrastive loss between generated samples and original image embeddings in the pre-trained diffusion model, our method can generate images with the same semantic content as the source image in a zero-shot manner. Our approach outperforms existing methods while preserving content and requiring no additional training, not only for image style transfer but also for image-to-image translation and manipulation. Our experimental results validate the effectiveness of our proposed method.


## How to Use
### Environment setting
**Python** 3.8.5 \
**Torch** 1.11.0 
```
$ conda env create -f environment.yml
$ conda activate zecon
```
Our source code relies on [blended diffusion](https://github.com/omriav/blended-diffusion).

### Pre-trained model
Download the model weights trained on [imagenet](https://github.com/openai/guided-diffusion) and [ffhq](https://github.com/jychoi118/ilvr_adm) dataset, respectively.

Create a folder ```'./ckpt/'``` and then place the downloaded weights into the folder.

### Image manipulation
In order to manipulate an image, run:
```
python main.py --output_path './results' --init_image './src_image/imagenet3.JPEG' --data 'imagenet' --prompt_tgt 'a sketch with crayon' --prompt_src 'Photo' \
--skip_timesteps 25 --timestep_respacing 50 --diffusion_type 'ddim_ddpm' --l_clip_global 0 --l_clip_global_patch 10000 --l_clip_dir 0 --l_clip_dir_patch 20000 \
--l_zecon 500 --l_mse 5000 --l_vgg 100 --patch_min 0.01 --patch_max 0.3
```
+ The path to the source image is given to the flag ```--init_image```
+ The flag ```--data``` indicates the pretrained diffusion model. If you manipulate face data, choose 'ffhq'.
+ The text prompt for the target style is given to the flag ```--prompt_tgt```
+ The text prompt for the style of the source image is given to the flag ```--prompt_src```
+ The flag ```--skip_timesteps``` indicates .
+ The flag ```--timestep_respacing``` indicates .
+ Diffusion sampling types are given to the flag ```--diffusion_type```. The first one is for the forward step, and the latter one is for the reverse step.


+ To further modulate the style, you can increase the four bottom losses.
  + The flag ```--l_clip_global``` indicates the weight for CLIP global loss.
  + The flag ```--l_clip_global_patch``` indicates the weight for patch-based CLIP global loss.
  + The flag ```--l_clip_dir``` indicates the weight for CLIP directional loss.
  + The flag ```--l_clip_dir_patch``` indicates the weight for patch-based CLIP directional loss.
+ To further preserve the content, you can increase the four bottom losses.
  + The flag ```--l_zecon``` indicates the weight for ZeCon loss.
  + The flag ```--l_mse``` indicates the weight for MSE loss.
  + The flag ```--l_vgg``` indicates the weight for VGG loss.
+ ***Tips!*** You can refer to the Table 5 in the [paper](https://arxiv.org/pdf/2303.08622.pdf) for the weights of the losses.


## BibTeX

```
@article{yang2023zero,
  title={Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer},
  author={Yang, Serin and Hwang, Hyunmin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2303.08622},
  year={2023}
}
```
