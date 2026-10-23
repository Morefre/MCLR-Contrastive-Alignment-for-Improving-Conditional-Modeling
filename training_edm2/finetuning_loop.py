# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

# #----------------------------------------------------------------------------------
# # Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss

# @persistence.persistent_class
# class MRLoss_ImageNet_EDM_Sampling_EDM_Weighting:
#     def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
#         self.P_mean = P_mean
#         self.P_std = P_std
#         self.sigma_data = sigma_data

#     def __call__(self, net, images, K=10, return_logvar=True, labels=None):
#         default_labels = torch.unique(labels, dim=0)
#         rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
#         sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#         weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
#         noise = torch.randn_like(images) * sigma
#         y_noisy = images + noise

#         if return_logvar == True:
#             D_yn, logvar = net(y_noisy, sigma, labels, return_logvar=True)
#         else:
#             D_yn = net(y_noisy, sigma, labels, return_logvar=False)

#         N = images.shape[0]

#         dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)

#         L_M = 0.00

#         all_neg_pairs = []
#         for i in range(N):
#             #print(labels)
#             c_i = labels[i]
#             mask_neg = (default_labels != c_i).any(dim=1)
#             #mask_neg[i] = False

#             neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
#             if neg_indices.numel() == 0:
#                 # No negative Sample found
#                 continue
            
#             # If there are more negative examples than K, random sample them
#             if neg_indices.numel() > K:
#                 perm = torch.randperm(neg_indices.numel(), device = images.device)
#                 neg_indices = neg_indices[perm[:K]]
            
#             for neg_j in neg_indices:
#                 all_neg_pairs.append((i, neg_j.item()))
        
#         M = len(all_neg_pairs)
#         #
#         if M == 0:
#             return 0.0

#         anchor_idx = []
#         class_idx = []
#         for (i,j) in all_neg_pairs:
#             anchor_idx.append(i)
#             class_idx.append(j)

#         anchor_idx = torch.tensor(anchor_idx, device=images.device, dtype=torch.long) 
#         class_idx  = torch.tensor(class_idx,  device=images.device, dtype=torch.long) 

#         labels_neg = default_labels[class_idx]
#         sigma_neg   = sigma[anchor_idx]
        
#         D_yn_wrong_all = net(y_noisy[anchor_idx], sigma_neg, labels_neg, return_logvar=False)
#         dist_wrong = (D_yn_wrong_all - images[anchor_idx])  # shape [M, C, H, W]
#         dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]

#         # dist_correct[ anchor_idx ] => shape [M]
#         # weight[ anchor_idx ] => shape [M, 1, 1, 1] or [M], depends on shape
#         # For simplicity, flatten weight to [N], then gather => [M]
#         w_anchor = weight.view(-1)[anchor_idx]  # shape [M]

#         # margin[k] = w_anchor[k]*(dist_correct[anchor_idx[k]] - dist_wrong[k])
#         if return_logvar == False:
#             margin_losses = w_anchor * (dist_correct[anchor_idx] - dist_wrong)
#         else:
#             margin_losses = (w_anchor / (logvar[anchor_idx].exp().reshape(w_anchor.shape))) * (dist_correct[anchor_idx] - dist_wrong) + logvar[anchor_idx].reshape(w_anchor.shape)

#         margin_losses = margin_losses.reshape(margin_losses.shape[0],-1)

#         L_M = margin_losses.sum() / float(N)

#         return L_M
#----------------------------------------------------------------------------------
# Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss
# Unifrom Sampling Uniform Weighting
@persistence.persistent_class
class MRLoss_ImageNet_Uniform_Sampling_Uniform_Weighting:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def __call__(self, net, images, K=10, return_logvar=True, labels=None):
        default_labels = torch.unique(labels, dim=0)
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min + rnd_uniform*(self.sigma_max-self.sigma_min)
        weight = torch.ones_like(sigma)
        noise = torch.randn_like(images) * sigma
        y_noisy = images + noise

        N = images.shape[0]

        L_M = 0.00

        all_neg_pairs = []
        for i in range(N):
            #print(labels)
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            #mask_neg[i] = False

            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue
            
            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))
        
        M = len(all_neg_pairs)
        #
        if M == 0:
            return 0.0

        anchor_idx = []
        class_idx = []
        for (i,j) in all_neg_pairs:
            anchor_idx.append(i)
            class_idx.append(j)

        anchor_idx = torch.tensor(anchor_idx, device=images.device, dtype=torch.long) 
        class_idx  = torch.tensor(class_idx,  device=images.device, dtype=torch.long) 

        labels_neg = default_labels[class_idx]
        sigma_neg   = sigma[anchor_idx]
        y_noisy_neg = y_noisy[anchor_idx]

        y_full = torch.cat([y_noisy, y_noisy_neg], dim=0)
        sigma_full = torch.cat([sigma, sigma_neg], dim=0)
        labels_full = torch.cat([labels, labels_neg], dim=0)

        D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)

        D_yn = D_yn_all[:y_noisy.shape[0]]
        D_yn_wrong_all = D_yn_all[y_noisy.shape[0]:]
        logvar = logvar_all[:y_noisy.shape[0]]
        
        dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
        dist_wrong = (D_yn_wrong_all - images[anchor_idx])  # shape [M, C, H, W]
        dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]

        assert dist_correct.shape[0] == images.shape[0]
        assert dist_wrong.shape[0] == len(anchor_idx)

        # dist_correct[ anchor_idx ] => shape [M]
        # weight[ anchor_idx ] => shape [M, 1, 1, 1] or [M], depends on shape
        # For simplicity, flatten weight to [N], then gather => [M]
        w_anchor = weight.view(-1)[anchor_idx]  # shape [M]

        # margin[k] = w_anchor[k]*(dist_correct[anchor_idx[k]] - dist_wrong[k])
        if return_logvar == False:
            margin_losses = w_anchor * (dist_correct[anchor_idx] - dist_wrong)
        else:
            margin_losses = (w_anchor / (logvar[anchor_idx].exp().reshape(w_anchor.shape))) * (dist_correct[anchor_idx] - dist_wrong) + logvar[anchor_idx].reshape(w_anchor.shape)

        margin_losses = margin_losses.reshape(margin_losses.shape[0],-1)

        L_M = margin_losses.sum() / float(N)

        return L_M
    

#----------------------------------------------------------------------------------
# Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss
# Unifrom Sampling EDM Weighting
@persistence.persistent_class
class MRLoss_ImageNet_Uniform_Sampling_EDM_Weighting:
    def __init__(self, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def __call__(self, net, images, K=10, return_logvar=True, labels=None):
        default_labels = torch.unique(labels, dim=0)
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min + rnd_uniform*(self.sigma_max-self.sigma_min)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        y_noisy = images + noise

        N = images.shape[0]

        L_M = 0.00

        all_neg_pairs = []
        for i in range(N):
            #print(labels)
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            #mask_neg[i] = False

            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue
            
            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))
        
        M = len(all_neg_pairs)
        #
        if M == 0:
            return 0.0

        anchor_idx = []
        class_idx = []
        for (i,j) in all_neg_pairs:
            anchor_idx.append(i)
            class_idx.append(j)

        anchor_idx = torch.tensor(anchor_idx, device=images.device, dtype=torch.long) 
        class_idx  = torch.tensor(class_idx,  device=images.device, dtype=torch.long) 

        labels_neg = default_labels[class_idx]
        sigma_neg   = sigma[anchor_idx]
        y_noisy_neg = y_noisy[anchor_idx]

        y_full = torch.cat([y_noisy, y_noisy_neg], dim=0)
        sigma_full = torch.cat([sigma, sigma_neg], dim=0)
        labels_full = torch.cat([labels, labels_neg], dim=0)

        D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)

        D_yn = D_yn_all[:y_noisy.shape[0]]
        D_yn_wrong_all = D_yn_all[y_noisy.shape[0]:]
        logvar = logvar_all[:y_noisy.shape[0]]
        
        dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
        dist_wrong = (D_yn_wrong_all - images[anchor_idx])  # shape [M, C, H, W]
        dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]

        assert dist_correct.shape[0] == images.shape[0]
        assert dist_wrong.shape[0] == len(anchor_idx)

        # dist_correct[ anchor_idx ] => shape [M]
        # weight[ anchor_idx ] => shape [M, 1, 1, 1] or [M], depends on shape
        # For simplicity, flatten weight to [N], then gather => [M]
        w_anchor = weight.view(-1)[anchor_idx]  # shape [M]

        # margin[k] = w_anchor[k]*(dist_correct[anchor_idx[k]] - dist_wrong[k])
        if return_logvar == False:
            margin_losses = w_anchor * (dist_correct[anchor_idx] - dist_wrong)
        else:
            margin_losses = (w_anchor / (logvar[anchor_idx].exp().reshape(w_anchor.shape))) * (dist_correct[anchor_idx] - dist_wrong) + logvar[anchor_idx].reshape(w_anchor.shape)

        margin_losses = margin_losses.reshape(margin_losses.shape[0],-1)

        L_M = margin_losses.sum() / float(N)

        return L_M
    
#----------------------------------------------------------------------------------
# Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss
# EDM Sampling with EDM Weighting

@persistence.persistent_class
class MRLoss_ImageNet_EDM_Sampling_EDM_Weighting:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, K=10, return_logvar=True, labels=None):
        default_labels = torch.unique(labels, dim=0)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        y_noisy = images + noise

        N = images.shape[0]

        L_M = 0.00

        all_neg_pairs = []
        for i in range(N):
            #print(labels)
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            #mask_neg[i] = False

            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue
            
            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))
        
        M = len(all_neg_pairs)
        #
        if M == 0:
            return 0.0

        anchor_idx = []
        class_idx = []
        for (i,j) in all_neg_pairs:
            anchor_idx.append(i)
            class_idx.append(j)

        anchor_idx = torch.tensor(anchor_idx, device=images.device, dtype=torch.long) 
        class_idx  = torch.tensor(class_idx,  device=images.device, dtype=torch.long) 

        labels_neg = default_labels[class_idx]
        sigma_neg   = sigma[anchor_idx]
        y_noisy_neg = y_noisy[anchor_idx]

        y_full = torch.cat([y_noisy, y_noisy_neg], dim=0)
        sigma_full = torch.cat([sigma, sigma_neg], dim=0)
        labels_full = torch.cat([labels, labels_neg], dim=0)

        D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)

        D_yn = D_yn_all[:y_noisy.shape[0]]
        D_yn_wrong_all = D_yn_all[y_noisy.shape[0]:]
        logvar = logvar_all[:y_noisy.shape[0]]
        
        dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
        dist_wrong = (D_yn_wrong_all - images[anchor_idx])  # shape [M, C, H, W]
        dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]

        assert dist_correct.shape[0] == images.shape[0]
        assert dist_wrong.shape[0] == len(anchor_idx)

        # dist_correct[ anchor_idx ] => shape [M]
        # weight[ anchor_idx ] => shape [M, 1, 1, 1] or [M], depends on shape
        # For simplicity, flatten weight to [N], then gather => [M]
        w_anchor = weight.view(-1)[anchor_idx]  # shape [M]

        # margin[k] = w_anchor[k]*(dist_correct[anchor_idx[k]] - dist_wrong[k])
        if return_logvar == False:
            margin_losses = w_anchor * (dist_correct[anchor_idx] - dist_wrong)
        else:
            margin_losses = (w_anchor / (logvar[anchor_idx].exp().reshape(w_anchor.shape))) * (dist_correct[anchor_idx] - dist_wrong) + logvar[anchor_idx].reshape(w_anchor.shape)

        margin_losses = margin_losses.reshape(margin_losses.shape[0],-1)

        L_M = margin_losses.sum() / float(N)

        return L_M

#----------------------------------------------------------------------------------
# Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss
# EDM Sampling with EDM Weighting

@persistence.persistent_class
class MRLoss_ImageNet_EDM_Sampling_Uniform_Weighting:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, K=10, return_logvar=True, labels=None):
        default_labels = torch.unique(labels, dim=0)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = torch.ones_like(sigma)
        noise = torch.randn_like(images) * sigma
        y_noisy = images + noise

        N = images.shape[0]

        L_M = 0.00

        all_neg_pairs = []
        for i in range(N):
            #print(labels)
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            #mask_neg[i] = False

            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue
            
            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))
        
        M = len(all_neg_pairs)
        #
        if M == 0:
            return 0.0

        anchor_idx = []
        class_idx = []
        for (i,j) in all_neg_pairs:
            anchor_idx.append(i)
            class_idx.append(j)

        anchor_idx = torch.tensor(anchor_idx, device=images.device, dtype=torch.long) 
        class_idx  = torch.tensor(class_idx,  device=images.device, dtype=torch.long) 

        labels_neg = default_labels[class_idx]
        sigma_neg   = sigma[anchor_idx]
        y_noisy_neg = y_noisy[anchor_idx]

        y_full = torch.cat([y_noisy, y_noisy_neg], dim=0)
        sigma_full = torch.cat([sigma, sigma_neg], dim=0)
        labels_full = torch.cat([labels, labels_neg], dim=0)

        D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)

        D_yn = D_yn_all[:y_noisy.shape[0]]
        D_yn_wrong_all = D_yn_all[y_noisy.shape[0]:]
        logvar = logvar_all[:y_noisy.shape[0]]
        
        dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
        dist_wrong = (D_yn_wrong_all - images[anchor_idx])  # shape [M, C, H, W]
        dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]

        assert dist_correct.shape[0] == images.shape[0]
        assert dist_wrong.shape[0] == len(anchor_idx)

        # dist_correct[ anchor_idx ] => shape [M]
        # weight[ anchor_idx ] => shape [M, 1, 1, 1] or [M], depends on shape
        # For simplicity, flatten weight to [N], then gather => [M]
        w_anchor = weight.view(-1)[anchor_idx]  # shape [M]

        # margin[k] = w_anchor[k]*(dist_correct[anchor_idx[k]] - dist_wrong[k])
        if return_logvar == False:
            margin_losses = w_anchor * (dist_correct[anchor_idx] - dist_wrong)
        else:
            margin_losses = (w_anchor / (logvar[anchor_idx].exp().reshape(w_anchor.shape))) * (dist_correct[anchor_idx] - dist_wrong) + logvar[anchor_idx].reshape(w_anchor.shape)

        margin_losses = margin_losses.reshape(margin_losses.shape[0],-1)

        L_M = margin_losses.sum() / float(N)

        return L_M

#----------------------------------------------------------------------------------
# Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss
# EDM Sampling with EDM Weighting

@persistence.persistent_class
class MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_Accumulated:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, negative_pairs_per_chunk=10):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.negative_pairs_per_chunk = negative_pairs_per_chunk

    def __call__(self, net, images, batch_gpu_total, K=10, return_logvar=True, labels=None, loss_scaling=1):
        default_labels = torch.unique(labels, dim=0)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        y_noisy = images + noise

        N = images.shape[0]

        L_M = 0.00

        all_neg_pairs = []
        for i in range(N):
            #print(labels)
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            #mask_neg[i] = False

            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue
            
            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))
        
        M = len(all_neg_pairs)

        anchor_idx = []
        class_idx = []
        for (i,j) in all_neg_pairs:
            anchor_idx.append(i)
            class_idx.append(j)

        num_chunks = (M + self.negative_pairs_per_chunk - 1) // self.negative_pairs_per_chunk

        anchor_idx = torch.tensor(anchor_idx, device=images.device, dtype=torch.long) 
        class_idx  = torch.tensor(class_idx,  device=images.device, dtype=torch.long) 

        total_loss_value = 0.0
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx*self.negative_pairs_per_chunk
            end_idx = min(start_idx+self.negative_pairs_per_chunk, M)
            anchor_idx_chunk = anchor_idx[start_idx:end_idx]
            class_idx_chunk = class_idx[start_idx:end_idx]

            labels_neg = default_labels[class_idx_chunk]
            y_noisy_neg = y_noisy[anchor_idx_chunk]
            sigma_neg   = sigma[anchor_idx_chunk]

            y_full = torch.cat([y_noisy_neg, y_noisy_neg], dim=0)
            sigma_full = torch.cat([sigma_neg, sigma_neg], dim=0)
            labels_full = torch.cat([labels[anchor_idx_chunk], labels_neg], dim=0)

            if chunk_idx < num_chunks - 1:
                with net.no_sync():
                    D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)
                    D_yn = D_yn_all[:y_noisy_neg.shape[0]]
                    D_yn_wrong_all = D_yn_all[y_noisy_neg.shape[0]:]
                    logvar = logvar_all[:y_noisy_neg.shape[0]]

                    dist_correct = (D_yn - images[anchor_idx_chunk]).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_wrong = (D_yn_wrong_all - images[anchor_idx_chunk])  # shape [M, C, H, W]
                    dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]
                    
                    assert dist_correct.shape[0] == images[anchor_idx_chunk].shape[0]
                    assert dist_wrong.shape[0] == len(anchor_idx_chunk)

                    w_anchor = weight.view(-1)[anchor_idx_chunk]

                    if return_logvar == False:
                        margin_losses = w_anchor * (dist_correct - dist_wrong)
                    else:
                        margin_losses =(w_anchor / (logvar.exp().reshape(w_anchor.shape))) * (dist_correct - dist_wrong) + logvar.reshape(w_anchor.shape)
                
                    margin_losses = margin_losses.reshape(margin_losses.shape[0],-1) * loss_scaling
                    L_M = margin_losses.sum() / (K*batch_gpu_total)
                    L_M.backward()
            
            else:
                D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)
                D_yn = D_yn_all[:y_noisy_neg.shape[0]]
                D_yn_wrong_all = D_yn_all[y_noisy_neg.shape[0]:]
                logvar = logvar_all[:y_noisy_neg.shape[0]]

                dist_correct = (D_yn - images[anchor_idx_chunk]).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_wrong = (D_yn_wrong_all - images[anchor_idx_chunk])  # shape [M, C, H, W]
                dist_wrong = dist_wrong.flatten(start_dim=1).pow(2).sum(dim=1)  # shape [M]
                
                assert dist_correct.shape[0] == images[anchor_idx_chunk].shape[0]
                assert dist_wrong.shape[0] == len(anchor_idx_chunk)

                w_anchor = weight.view(-1)[anchor_idx_chunk]

                if return_logvar == False:
                    margin_losses = w_anchor * (dist_correct - dist_wrong)
                else:
                    margin_losses =(w_anchor / (logvar.exp().reshape(w_anchor.shape))) * (dist_correct - dist_wrong) + logvar.reshape(w_anchor.shape)
            
                margin_losses = margin_losses.reshape(margin_losses.shape[0],-1) * loss_scaling
                L_M = margin_losses.sum() / (K*batch_gpu_total)
                L_M.backward()

            total_loss_value  = total_loss_value + float(L_M.detach())

            del D_yn, D_yn_wrong_all, dist_correct, dist_wrong, margin_losses # delete used variables to save space

        return total_loss_value


#----------------------------------------------------------------------------------
# Uncertainty-based loss function for calculating the Maximum likelihood-ratio loss
# EDM Sampling with EDM Weighting
# Simplified version, using torch.roll to build negative labels

@persistence.persistent_class
class MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_simplified:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def __call__(self, net, images, batch_gpu_total, beta=1, return_logvar=True, labels=None, loss_scaling=1):
        if self.P_std < 0:
            # If P_std <0, use uniform sampling between sigma_min and sigma_max
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            sigma = self.sigma_min + rnd_uniform*(self.sigma_max-self.sigma_min)
        else:
            rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        y_noisy = images + noise

        N = images.shape[0]

        L_M = 0.00

        labels_neg = torch.roll(labels, shifts=-1, dims=0)
        y_noisy_full = torch.cat([y_noisy, y_noisy], dim=0)
        sigma_full = torch.cat([sigma, sigma], dim=0)
        labels_full = torch.cat([labels, labels_neg], dim=0)

        if return_logvar == True:
            D_yn_all, logvar_all = net(y_noisy_full, sigma_full, labels_full, return_logvar=True)
            D_yn = D_yn_all[:y_noisy.shape[0]]
            D_yn_wrong = D_yn_all[y_noisy.shape[0]:]
            logvar = logvar_all[:y_noisy.shape[0]]

            dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
            dist_wrong = (D_yn_wrong - images).flatten(start_dim=1).pow(2).sum(dim=1)
            
            margin_losses = (weight / (logvar.exp().reshape(weight.shape))) * (dist_correct - beta*dist_wrong) + logvar.reshape(weight.shape)

        elif return_logvar == False:
            D_yn_all = net(y_noisy_full, sigma_full, labels_full, return_logvar=False)
            D_yn = D_yn_all[:y_noisy.shape[0]]
            D_yn_wrong = D_yn_all[y_noisy.shape[0]:]

            dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
            dist_wrong = (D_yn_wrong - images).flatten(start_dim=1).pow(2).sum(dim=1)
            
            margin_losses = weight * (dist_correct - beta*dist_wrong)

        margin_losses = margin_losses.reshape(margin_losses.shape[0],-1) * loss_scaling
        L_M = margin_losses.sum() / batch_gpu_total
        L_M.backward()

        return float(L_M.detach())

#----------------------------------------------------------------------------------
# DPO finetuning

@persistence.persistent_class
class DPO_Loss_EDM_Sampling_EDM_Weighting_Accumulated:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, negative_pairs_per_chunk=10):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.negative_pairs_per_chunk = negative_pairs_per_chunk

    def __call__(self, net, net_ref, images, batch_gpu_total, beta=0.1, K=10, return_logvar=True, labels=None, loss_scaling=1):
        default_labels = labels
        N = images.shape[0]
        
        L_M = 0.00
        all_neg_pairs = []
        for i in range(N):
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue

            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))

        M = len(all_neg_pairs)

        winning_idx = []
        losing_idx = []
        for (i,j) in all_neg_pairs:
            winning_idx.append(i)
            losing_idx.append(j)
        
        winning_idx = torch.tensor(winning_idx, device=images.device, dtype=torch.long) 
        losing_idx  = torch.tensor(losing_idx,  device=images.device, dtype=torch.long) 

        num_chunks = (M + self.negative_pairs_per_chunk - 1) // self.negative_pairs_per_chunk

        total_loss_value = 0.0
    
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx*self.negative_pairs_per_chunk 
            end_idx = min(start_idx+self.negative_pairs_per_chunk, M)

            winning_idx_chunk = winning_idx[start_idx:end_idx]
            losing_idx_chunk = losing_idx[start_idx:end_idx]
            
            labels_winning = default_labels[winning_idx_chunk]
            y_winning = images[winning_idx_chunk]
            y_losing = images[losing_idx_chunk]

            rnd_normal = torch.randn([y_winning.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

            noise = torch.randn_like(y_winning) * sigma
            y_noisy_winning = y_winning + noise
            y_noisy_losing = y_losing + noise

            y_full = torch.cat([y_noisy_winning, y_noisy_losing], dim=0)
            sigma_full = torch.cat([sigma, sigma], dim=0)
            labels_full = torch.cat([labels_winning, labels_winning], dim=0)

            if chunk_idx < num_chunks-1:
                retain_graph = True
                with net.no_sync():
                    '''
                    Again, here, net cannot be called twice.
                    '''
                    #print(y_full.shape, sigma_full.shape, labels_full.shape)
                    D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)
                    D_yn_model_winning = D_yn_all[:y_noisy_winning.shape[0]]
                    D_yn_model_losing =  D_yn_all[y_noisy_winning.shape[0]:]
                    logvar = logvar_all[:y_noisy_winning.shape[0]]

                    D_yn_ref_winning = net_ref(y_noisy_winning, sigma, labels_winning, return_logvar=False)
                    D_yn_ref_losing = net_ref(y_noisy_losing, sigma, labels_winning, return_logvar=False)

                    dist_model_winning = (D_yn_model_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_ref_winning = (D_yn_ref_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_model_losing = (D_yn_model_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_ref_losing = (D_yn_ref_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)

                    winning_diff = dist_model_winning - dist_ref_winning
                    losing_diff = dist_model_losing - dist_ref_losing

                    L_M = -torch.nn.functional.logsigmoid(-1*beta*weight*(winning_diff - losing_diff))
                    if return_logvar == True:
                        L_M = L_M / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)
                    L_M = L_M.sum() / (K*batch_gpu_total) * loss_scaling
                    L_M.backward(retain_graph=retain_graph)

                    # if return_logvar == False:
                    #     inside_term = torch.nn.functional.logsigmoid(-1*beta*weight*(winning_diff - losing_diff))
                    # else:
                    #     #inside_term = torch.nn.functional.logsigmoid(-1*beta*weight/ (logvar.exp().reshape(weight.shape)) * (winning_diff - losing_diff))
                    #     '''
                    #     To do: fix this
                    #     '''
                    #     #inside_term = torch.nn.functional.logsigmoid(-1*beta*weight*(winning_diff - losing_diff)) / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)

                    # L_M = -inside_term.sum()/(K*batch_gpu_total)*loss_scaling

            else:
                retain_graph = False
                #print(y_full.shape, sigma_full.shape, labels_full.shape)
                D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)
                D_yn_model_winning = D_yn_all[:y_noisy_winning.shape[0]]
                D_yn_model_losing =  D_yn_all[y_noisy_winning.shape[0]:]
                logvar = logvar_all[:y_noisy_winning.shape[0]]

                D_yn_ref_winning = net_ref(y_noisy_winning, sigma, labels_winning, return_logvar=False)
                D_yn_ref_losing = net_ref(y_noisy_losing, sigma, labels_winning, return_logvar=False)

                dist_model_winning = (D_yn_model_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_ref_winning = (D_yn_ref_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_model_losing = (D_yn_model_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_ref_losing = (D_yn_ref_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)

                winning_diff = dist_model_winning - dist_ref_winning
                losing_diff = dist_model_losing - dist_ref_losing
                
                L_M = -torch.nn.functional.logsigmoid(-1*beta*weight*(winning_diff - losing_diff))
                if return_logvar == True:
                    L_M = L_M / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)
                L_M = L_M / (K*batch_gpu_total) * loss_scaling
                L_M = L_M.sum()
                L_M.backward(retain_graph=retain_graph)

            total_loss_value = total_loss_value + float(L_M.detach())

        return total_loss_value

#----------------------------------------------------------------------------------
# CCA finetuning

@persistence.persistent_class
class CCA_Loss_EDM_Sampling_EDM_Weighting_Accumulated:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, negative_pairs_per_chunk=10):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.negative_pairs_per_chunk = negative_pairs_per_chunk

    def __call__(self, net, net_ref, images, batch_gpu_total, beta=0.1, lamb=1,  K=10, return_logvar=True, labels=None, loss_scaling=1):
        default_labels = labels
        N = images.shape[0]
        
        L_M = 0.00
        all_neg_pairs = []
        for i in range(N):
            c_i = labels[i]
            mask_neg = (default_labels != c_i).any(dim=1)
            neg_indices = mask_neg.nonzero().squeeze() # get all valid indices
            if neg_indices.numel() == 0:
                # No negative Sample found
                continue

            # If there are more negative examples than K, random sample them
            if neg_indices.numel() > K:
                perm = torch.randperm(neg_indices.numel(), device = images.device)
                neg_indices = neg_indices[perm[:K]]
            
            for neg_j in neg_indices:
                all_neg_pairs.append((i, neg_j.item()))

        M = len(all_neg_pairs)

        winning_idx = []
        losing_idx = []
        for (i,j) in all_neg_pairs:
            winning_idx.append(i)
            losing_idx.append(j)
        
        winning_idx = torch.tensor(winning_idx, device=images.device, dtype=torch.long) 
        losing_idx  = torch.tensor(losing_idx,  device=images.device, dtype=torch.long) 

        num_chunks = (M + self.negative_pairs_per_chunk - 1) // self.negative_pairs_per_chunk

        total_loss_value = 0.0
    
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx*self.negative_pairs_per_chunk 
            end_idx = min(start_idx+self.negative_pairs_per_chunk, M)

            winning_idx_chunk = winning_idx[start_idx:end_idx]
            losing_idx_chunk = losing_idx[start_idx:end_idx]
            
            labels_winning = default_labels[winning_idx_chunk]
            y_winning = images[winning_idx_chunk]
            y_losing = images[losing_idx_chunk]

            rnd_normal = torch.randn([y_winning.shape[0], 1, 1, 1], device=images.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

            noise = torch.randn_like(y_winning) * sigma
            y_noisy_winning = y_winning + noise
            y_noisy_losing = y_losing + noise

            y_full = torch.cat([y_noisy_winning, y_noisy_losing], dim=0)
            sigma_full = torch.cat([sigma, sigma], dim=0)
            labels_full = torch.cat([labels_winning, labels_winning], dim=0)

            if chunk_idx < num_chunks-1:
                retain_graph = True
                with net.no_sync():
                    '''
                    Again, here, net cannot be called twice.
                    '''
                    #print(y_full.shape, sigma_full.shape, labels_full.shape)
                    D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)
                    D_yn_model_winning = D_yn_all[:y_noisy_winning.shape[0]]
                    D_yn_model_losing =  D_yn_all[y_noisy_winning.shape[0]:]
                    logvar = logvar_all[:y_noisy_winning.shape[0]]

                    D_yn_ref_winning = net_ref(y_noisy_winning, sigma, labels_winning, return_logvar=False)
                    D_yn_ref_losing = net_ref(y_noisy_losing, sigma, labels_winning, return_logvar=False)

                    dist_model_winning = (D_yn_model_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_ref_winning = (D_yn_ref_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_model_losing = (D_yn_model_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)
                    dist_ref_losing = (D_yn_ref_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)

                    winning_diff = dist_model_winning - dist_ref_winning
                    losing_diff = dist_model_losing - dist_ref_losing

                    L_M = torch.nn.functional.logsigmoid(-1*beta*weight*winning_diff) + lamb*torch.nn.functional.logsigmoid(beta*weight*losing_diff)
                    L_M = - L_M
                    if return_logvar == True:
                        L_M = L_M / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)
                    L_M = L_M.sum() / (K*batch_gpu_total) * loss_scaling
                    L_M.backward(retain_graph=retain_graph)

                    # if return_logvar == False:
                    #     inside_term = torch.nn.functional.logsigmoid(-1*beta*weight*(winning_diff - losing_diff))
                    # else:
                    #     #inside_term = torch.nn.functional.logsigmoid(-1*beta*weight/ (logvar.exp().reshape(weight.shape)) * (winning_diff - losing_diff))
                    #     '''
                    #     To do: fix this
                    #     '''
                    #     #inside_term = torch.nn.functional.logsigmoid(-1*beta*weight*(winning_diff - losing_diff)) / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)

                    # L_M = -inside_term.sum()/(K*batch_gpu_total)*loss_scaling

            else:
                retain_graph = False
                #print(y_full.shape, sigma_full.shape, labels_full.shape)
                D_yn_all, logvar_all = net(y_full, sigma_full, labels_full, return_logvar=return_logvar)
                D_yn_model_winning = D_yn_all[:y_noisy_winning.shape[0]]
                D_yn_model_losing =  D_yn_all[y_noisy_winning.shape[0]:]
                logvar = logvar_all[:y_noisy_winning.shape[0]]

                D_yn_ref_winning = net_ref(y_noisy_winning, sigma, labels_winning, return_logvar=False)
                D_yn_ref_losing = net_ref(y_noisy_losing, sigma, labels_winning, return_logvar=False)

                dist_model_winning = (D_yn_model_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_ref_winning = (D_yn_ref_winning - y_winning).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_model_losing = (D_yn_model_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)
                dist_ref_losing = (D_yn_ref_losing - y_losing).flatten(start_dim=1).pow(2).sum(dim=1)

                winning_diff = dist_model_winning - dist_ref_winning
                losing_diff = dist_model_losing - dist_ref_losing
                
                L_M = torch.nn.functional.logsigmoid(-1*beta*weight*winning_diff) + lamb*torch.nn.functional.logsigmoid(beta*weight*losing_diff)
                L_M = -L_M
                if return_logvar == True:
                    L_M = L_M / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)
                L_M = L_M / (K*batch_gpu_total) * loss_scaling
                L_M = L_M.sum()
                L_M.backward(retain_graph=retain_graph)

            total_loss_value = total_loss_value + float(L_M.detach())

        return total_loss_value


#----------------------------------------------------------------------------------
# CCA finetuning

@persistence.persistent_class
class CCA_Loss_EDM_Sampling_EDM_Weighting_simplified:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, net_ref, images, batch_gpu_total, beta=0.1, lamb=1, return_logvar=True, labels=None, loss_scaling=1):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        
        y_neg = torch.roll(images, shifts=-1, dims=0)
        y_noisy = images + noise
        y_noisy_neg = y_neg + noise

        N = images.shape[0]
        
        L_M = 0.00

        y_noisy_full = torch.cat([y_noisy, y_noisy_neg], dim=0)
        sigma_full = torch.cat([sigma, sigma], dim=0)
        labels_full = torch.cat([labels, labels], dim=0)

        D_yn_ref_correct = net_ref(y_noisy, sigma, labels, return_logvar=False)
        D_yn_ref_wrong = net_ref(y_noisy_neg, sigma, labels, return_logvar=False)
        dist_ref_correct = (D_yn_ref_correct - images).flatten(start_dim=1).pow(2).sum(dim=1)
        dist_ref_wrong = (D_yn_ref_wrong - y_neg).flatten(start_dim=1).pow(2).sum(dim=1)
        
        '''
        continue here
        '''

        if return_logvar == True:
            D_yn_all, logvar_all = net(y_noisy_full, sigma_full, labels_full, return_logvar=True)
            D_yn = D_yn_all[:y_noisy.shape[0]]
            D_yn_wrong = D_yn_all[y_noisy.shape[0]:]
            logvar = logvar_all[:y_noisy.shape[0]]

            dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
            dist_wrong = (D_yn_wrong - images).flatten(start_dim=1).pow(2).sum(dim=1)

            winning_diff = dist_correct - dist_ref_correct
            losing_diff = dist_wrong - dist_ref_wrong

            L_M = torch.nn.functional.logsigmoid(-1*beta*weight*winning_diff) + lamb*torch.nn.functional.logsigmoid(beta*weight*losing_diff)
            L_M = - L_M
            L_M = L_M / (logvar.exp().reshape(weight.shape)) + logvar.reshape(weight.shape)
                        
        elif return_logvar == False:
            D_yn_all = net(y_noisy_full, sigma_full, labels_full, return_logvar=False)
            D_yn = D_yn_all[:y_noisy.shape[0]]
            D_yn_wrong = D_yn_all[y_noisy.shape[0]:]

            dist_correct = (D_yn - images).flatten(start_dim=1).pow(2).sum(dim=1)
            dist_wrong = (D_yn_wrong - images).flatten(start_dim=1).pow(2).sum(dim=1)

            winning_diff = dist_correct - dist_ref_correct
            losing_diff = dist_wrong - dist_ref_wrong

            L_M = torch.nn.functional.logsigmoid(-1*beta*weight*winning_diff) + lamb*torch.nn.functional.logsigmoid(beta*weight*losing_diff)
            L_M = - L_M

        L_M = L_M.reshape(L_M.shape[0],-1) * loss_scaling
        L_M = L_M.sum() / batch_gpu_total
        L_M.backward()


        return float(L_M.detach())

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def finetuning_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),
    reference_net_pkl   = None,     # Path to the weights pkl of the reference network.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    k                   = 10,       # Number of negative pairs
    beta                = 1,      # beta for DPO KL divergence
    lamb                = 1,        # lambda for CCA    
    return_logvar       = True,      # Whether use dynamic weighting when computing training loss
    use_decay           = False,
    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    torch.autograd.set_detect_anomaly(True)
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    #assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0)
    #assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    #assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net = net.to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    if return_logvar == False:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device],find_unused_parameters=True)
    else:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])

    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    #ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        #misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory

    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    #checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    net.train().requires_grad_(True).to(device)

    # Initialize reference network.
    #assert reference_net_pkl != None
    # Construct reference network
    if reference_net_pkl != 'None' and reference_net_pkl != None:
        dist.print0('Constructing reference network')
        interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
        net_ref = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
        net_ref.eval().requires_grad_(False).to(device)

        dist.print0(f'Loading reference network weights from "{reference_net_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(reference_net_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net_ref, require_all=False)
        del data # conserve memory
    else:
        net_ref = None

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                if loss_kwargs.class_name == 'training.finetuning_loop.MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_Accumulated':
                    loss = loss_fn(net=ddp, images=images, batch_gpu_total=batch_gpu_total, K=k, return_logvar = return_logvar, labels=labels.to(device), loss_scaling=loss_scaling)
                elif loss_kwargs.class_name == 'training.finetuning_loop.MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_simplified':
                    loss = loss_fn(net=ddp, images=images, batch_gpu_total=batch_gpu_total, beta=beta, return_logvar = return_logvar, labels=labels.to(device), loss_scaling=loss_scaling)
                elif loss_kwargs.class_name == 'training.finetuning_loop.DPO_Loss_EDM_Sampling_EDM_Weighting_Accumulated':
                    loss = loss_fn(net=ddp, net_ref=net_ref, images=images, batch_gpu_total=batch_gpu_total, beta=beta, K=k, return_logvar = return_logvar, labels=labels.to(device), loss_scaling=loss_scaling)
                elif loss_kwargs.class_name == 'training.finetuning_loop.CCA_Loss_EDM_Sampling_EDM_Weighting_Accumulated':
                    loss = loss_fn(net=ddp, net_ref=net_ref, images=images, batch_gpu_total=batch_gpu_total, beta=beta, lamb=lamb, K=k, return_logvar = return_logvar, labels=labels.to(device), loss_scaling=loss_scaling)
                elif loss_kwargs.class_name == 'training.finetuning_loop.CCA_Loss_EDM_Sampling_EDM_Weighting_simplified':
                    loss = loss_fn(net=ddp, net_ref=net_ref, images=images, batch_gpu_total=batch_gpu_total, beta=beta, lamb=lamb, return_logvar = return_logvar, labels=labels.to(device), loss_scaling=loss_scaling)
                else:
                    loss = loss_fn(net=ddp, images=images, labels=labels.to(device), K=k, return_logvar = return_logvar)
                    loss.mul(loss_scaling).backward()

                training_stats.report('Loss/loss', loss)
                #loss.sum().mul(loss_scaling / batch_gpu_total).backward()


        # Run optimizer and update weights.
        if use_decay == True:
            lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        else:
            lr = lr_kwargs['ref_lr']

        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
           g['lr'] = lr
        
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

#----------------------------------------------------------------------------
