# PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn

def d_clip_loss(x, y, use_cosine=False):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance

def d_clip_dir_loss(x_embd,y_embd,prompt_x_embd,prompt_y_embd):
    d_img = x_embd - y_embd
    d_txt = prompt_x_embd - prompt_y_embd

    d_img = F.normalize(d_img, dim=-1)
    d_txt = F.normalize(d_txt, dim=-1)
    
    distance = 1 - (d_img @ d_txt.t()).squeeze()
    
    return distance

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def mse_loss(x_in, y_in):
    mse = torch.nn.MSELoss()
    return mse(x_in,y_in)

def get_features(image, model, layers=None):
    
    if layers is None:
        layers = {'0': 'conv1_1', 
                  '2': 'conv1_2', 
                  '5': 'conv2_1',  
                  '7': 'conv2_2',
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def zecon_loss_direct(Unet, x_in, y_in,t):
    total_loss = 0
    nce_layers = [0,2,5,8,11]
    num_patches=256

    l2norm = Normalize(2)
    feat_q = Unet.forward_enc(x_in,t, nce_layers)
    feat_k = Unet.forward_enc(y_in,t, nce_layers)
    patch_ids = []
    feat_k_pool = []
    feat_q_pool = []
    
    for feat_id, feat in enumerate(feat_k):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)       # [B,ch,h,w] > [B,h*w,ch]

        patch_id = np.random.permutation(feat_reshape.shape[1])
        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        
        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        
        patch_ids.append(patch_id)
        x_sample = l2norm(x_sample)
        feat_k_pool.append(x_sample)
    
    for feat_id, feat in enumerate(feat_q):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)       # [B,ch,h,w] > [B,h*w,ch]

        patch_id = patch_ids[feat_id]

        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        
        x_sample = l2norm(x_sample)
        feat_q_pool.append(x_sample)

    for f_q, f_k in zip(feat_q_pool, feat_k_pool):
        loss = PatchNCELoss(f_q, f_k)
        total_loss += loss.mean()
    return total_loss.mean()

def PatchNCELoss(feat_q, feat_k, batch_size=1, nce_T = 0.07):
    # feat_q : n_patch x 512
    # feat_q : n_patch x 512
    batch_size = batch_size
    nce_T = nce_T
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    mask_dtype = torch.bool

    num_patches = feat_q.shape[0]
    dim = feat_q.shape[1]
    feat_k = feat_k.detach()
    
    # pos logit 
    l_pos = torch.bmm(
        feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
    l_pos = l_pos.view(num_patches, 1)

    # reshape features to batch size
    feat_q = feat_q.view(batch_size, -1, dim)
    feat_k = feat_k.view(batch_size, -1, dim)
    npatches = feat_q.size(1)
    l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

    # diagonal entries are similarity between same features, and hence meaningless.
    # just fill the diagonal with very small number, which is exp(-10) and almost zero
    diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
    l_neg_curbatch.masked_fill_(diagonal, -10.0)
    l_neg = l_neg_curbatch.view(-1, npatches)

    out = torch.cat((l_pos, l_neg), dim=1) / nce_T

    loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                    device=feat_q.device))

    return loss




