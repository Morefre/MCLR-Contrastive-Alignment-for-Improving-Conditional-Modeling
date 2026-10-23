# MCLR

## 1. Preparing the Environment and Training Datasets
To set up the environment and prepare training datasets, please refer to [EDM2](https://github.com/NVlabs/edm2).

## 2. Training New Models
The complete training commands are provided in [run.sh](https://github.com/Morefre/MCLR/blob/MCLR-for-EDM2/edm2/run.sh). 

We recommend using the "Simplified version" for faster training and the 'Accumulated version' for better performance.


## 3. Model Evaluation

Before evaluating the model, we need to sample 50k images with the following command:

```
CKPT="/path/to/network-snapshot-xxxxxxx-xxx.pkl"
OUTDIR="generation_ImageNet/mclr_model"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    generate_images.py \
    --net="$CKPT" \
    --outdir="$OUTDIR" \
    --seeds=0-49999 \
    --batch=32 \
    --steps=32 \
    --guidance=1 \
    --subdirs
```

Here guidance=1 implies no CFG is used.

### 1) Calculate FID and FD<sub>DINOv2</sub>
To calculate FID and FD<sub>DINOv2</sub> scores, we first need to compute the reference statistics. Here we use 256×256 as an example. The same procedure applies to other resolutions.

```
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
cd edm2
python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/img256.zip --resolution=256x256 --transform=center-crop-dhariwal 
```

Next, compute reference statistics:

```
python calculate_metrics.py ref --data=datasets/img256.zip --dest=fd-refs/img_256.pkl 
```

Now let's compute the FID and FD_DinoV2 scores by running the following code, where "out" is the folder in which you store 50,000 generated images:

```
torchrun --nproc_per_node=4 calculate_metrics.py calc --images=out \
          --save_path=FD_scores --num=50000 --ref=fd-refs/img_256.pkl 
```

### 2) Compute Precision and Recall

Both Precision and Recall can be computed on both DinoV2 and Inception features, hence we need to prepare reference statistics for both.
To do so, we first need to sample 50,000 images from the training dataset. Since ImageNet has 1000 calss, we need to sample 50 samples from each class:

```
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution, randomly selecting 50 samples for each of the 1000 class. The final dataset will have 50,000 images in total.
cd edm2
python dataset_tool_50k.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/img256_50k.zip --resolution=256x256 --transform=center-crop-dhariwal 
```

```
# Prepare features for calculating precision and recall.
torchrun --nproc_per_node=1 calculate_metrics.py calc-features  --images=datasets/img256_50k.zip \
                                                                 --save_path=feature_embeddings_ImageNet/img256_50k
```

Now, let's compute the Precision and Recall:

```
torchrun --nproc_per_node=4 calculate_metrics.py calc-precision-recall --images=out \
         --save_path=Precision_and_recall/ --ref_feature_path=feature_embeddings_ImageNet/img256_50k
```

### 3) Compute Inception Score

Computing Inception score is easier, since we don't need to prepare reference statistics in advance. All we need is a collection of 50,000 generated images stored in folder '"out":

```
torchrun --nproc_per_node=4 calculate_metrics.py calc-inception  --images=out --save_path=Inception_score/

```

## 4. References
Our implementation is based on [EDM2](https://github.com/NVlabs/edm2).
