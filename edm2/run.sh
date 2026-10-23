
##################################
# dataset: ImageNet 512x512
# base model: EDM2-L
# base FD_Dinov2: 67.70
##################################

# All experiments are conducted on a single node with 4 A100 GPUs (80 GB each). 
# If computational resources are limited, we recommend (i) switching to the simplified version, and (ii) reducing the batch-gpu or the negative_pairs_per_chunk.

# MCLR Accumulated version
# Optimal FD_Dinov2: 42.50 at checkpoint 896
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_512_MCLR_Accumulated \
    --data=datasets/img512-sd.zip --cond=True --preset=edm2-img512-l-MCLR-accumulated \
    --batch=560 --lr=1e-5 --decay=35000 --k=10 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --batch-gpu=10 --fp16=False --use_decay=False --P_std=3.0 --negative_pairs_per_chunk=25


# CC-DPO Accumulated version
# Optimal FD_Dinov2: 51.92 at checkpoint 1904
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_512_DPO_Accumulated \
    --data=datasets/img512-sd.zip --cond=True --preset=edm2-img512-l-DPO-accumulated \
    --batch=560 --lr=1e-4 --beta=0.1 --decay=35000 --k=10 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --batch-gpu=10 --fp16=False --use_decay=False --P_std=3.0 --negative_pairs_per_chunk=15


# CCA Accumulated version
# Optimal FD_Dinov2: 51.45 at checkpoint 1904
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_512_CCA_Accumulated \
    --data=datasets/img512-sd.zip --cond=True --preset=edm2-img512-l-CCA-accumulated \
    --batch=560 --lr=1e-4 --beta=0.1 --decay=35000 --k=10 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --batch-gpu=10 --fp16=False --use_decay=False --P_std=3.0 --negative_pairs_per_chunk=15 --lamb=10


# The accumulated versions are computationally demanding.
# For limited resources, use the simplified versions below.
# We reuse hyperparameters from the accumulated versions without retuning,
# so performance may slightly degrade, but training is faster and more memory-efficient.

# MCLR Simplified version
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_512_MCLR_Simplified \
    --data=datasets/img512-sd.zip --cond=True --preset=edm2-img512-l-MCLR-simplified \
    --batch=560 --lr=1e-5 --decay=35000 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --batch-gpu=35 --fp16=False --use_decay=False --P_std=3.0


# CC-DPO Simplified version
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_512_DPO_Simplified \
    --data=datasets/img512-sd.zip --cond=True --preset=edm2-img512-l-DPO-simplified \
    --batch=560 --lr=1e-4 --beta=0.1 --decay=35000 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --batch-gpu=35 --fp16=False --use_decay=False --P_std=3.0


# CCA Simplified version
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_512_CCA_Simplified \
    --data=datasets/img512-sd.zip --cond=True --preset=edm2-img512-l-CCA-simplified \
    --batch=560 --lr=1e-4 --beta=0.1 --decay=35000 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl'\
    --batch-gpu=35 --fp16=False --use_decay=False --P_std=3.0 --lamb=10


##################################
# dataset: ImageNet 64x64
# base model: EDM2-S
# base FD_Dinov2: 95.20
##################################

# All experiments are conducted on a single node with 4 A100 GPUs (80 GB each). 
# If computational resources are limited, we recommend (i) switching to the simplified version, and (ii) reducing the batch-gpu or the negative_pairs_per_chunk.

# MCLR Accumulated version
# Optimal FD_Dinov2: 52.69 at checkpoint 392
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_MCLR_Accumulated \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-MCLR-accumulated \
    --batch=560 --lr=1e-5 --decay=35000 --k=10 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=10 --fp16=False --use_decay=False --P_std=3.0 --negative_pairs_per_chunk=25

# CC-DPO Accumulated version
# Optimal FD_Dinov2: 60.98 at checkpoint 2016
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_DPO_Accumulated \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-DPO-accumulated \
    --batch=560 --lr=1e-4 --beta=0.001 --decay=35000 --k=10 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=10 --fp16=False --use_decay=False --P_std=3.0 --negative_pairs_per_chunk=15


# CCA Accumulated version
# Optimal FD_Dinov2: 62.36 at checkpoint 672
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_CCA_Accumulated \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-CCA-accumulated \
    --batch=560 --lr=1e-4 --beta=0.001 --decay=35000 --k=10 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=10 --fp16=False --use_decay=False --P_std=3.0 --negative_pairs_per_chunk=15 --lamb=10

# The accumulated versions are computationally demanding.
# For limited resources, use the simplified versions below.
# We reuse hyperparameters from the accumulated versions without retuning,
# so performance may slightly degrade, but training is faster and more memory-efficient.

# MCLR Simplified version, without DSM term

torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_MCLR_Simplified \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-MCLR-simplified \
    --batch=560 --lr=1e-4 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=35 --fp16=False --use_decay=False --P_std=3.0

# MCLR Simplified version, with DSM term (discussed in appendix G.1, figure 13)
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_MCLR_Simplified_DSM \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-MCLR-simplified-DSM \
    --batch=560 --lr=1e-4 --return_logvar=True --status=5600 --snapshot=5600 --checkpoint=560000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=7 --fp16=False --use_decay=False --P_std=3.0 --beta=100

# CC-DPO Simplified version
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_DPO_Simplified \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-DPO-simplified \
    --batch=560 --lr=1e-4 --beta=0.001 --decay=35000 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=35 --fp16=False --use_decay=False --P_std=3.0


# CCA Simplified version
torchrun --nproc_per_node=4 finetuning_edm2.py --outdir=training-runs/ImageNet_64_CCA_Simplified \
    --data=datasets/imagenet-64x64.zip --cond=True --preset=edm2-img64-s-CCA-simplified \
    --batch=560 --lr=1e-4 --beta=0.001 --decay=35000 --return_logvar=True --status=56000 --snapshot=56000 --checkpoint=5600000\
    --resume_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --reference_net_pkl='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'\
    --batch-gpu=35 --fp16=False --use_decay=False --P_std=3.0 --lamb=10