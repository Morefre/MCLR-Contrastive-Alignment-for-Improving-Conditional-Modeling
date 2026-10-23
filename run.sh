# Best observed results among evaluated checkpoints; iterations are optimizer steps.
# Evaluation: 50-step Euler ODE, 50,000 images, model weights.
# Fine-tuned models below are evaluated with CFG=1.

# REPA base model references (iteration 0; same checkpoint for all CFG scales):
# Without CFG (scale=1.0): FID=8.21908, FD_DINOv2=184.361.
# With CFG (scale=1.4; best observed FID): FID=2.08756, FD_DINOv2=80.9318.
# With CFG (scale=2.0; best observed FD_DINOv2): FID=4.59677, FD_DINOv2=50.2323.

# CFM + DSM (reg=1)
# Best FID: 4.52152 at iteration 100.
# Best FD_DINOv2: 114.941 at iteration 100.
accelerate launch --num_processes 1 MCLR_finetuning.py --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --encoder-depth=8 \
  --enc-type="None" \
  --output-dir="exps_CFM" \
  --exp-name="cfm_finetuning_dsm_reg1_ema_0.9999_lr_1e-6" \
  --data-dir="data" \
  --batch-size=64 \
  --max-train-steps=10000 \
  --checkpointing-steps=100 \
  --sampling-steps=50 \
  --learning-rate=1e-6 \
  --ema-weight-decay=0.9999 \
  --initial-weight="pretrained_models/last.pt" \
  --loss-type="CFM" \
  --use-dsm \
  --dsm-reg=1

# CFM + DSM (reg=5)
# Best FID: 4.89362 at iteration 200.
# Best FD_DINOv2: 115.664 at iteration 300.
accelerate launch --num_processes 1 MCLR_finetuning.py --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --encoder-depth=8 \
  --enc-type="None" \
  --output-dir="exps_CFM" \
  --exp-name="cfm_finetuning_dsm_reg5_ema_0.9999_lr_1e-6" \
  --data-dir="data" \
  --batch-size=64 \
  --max-train-steps=10000 \
  --checkpointing-steps=100 \
  --sampling-steps=50 \
  --learning-rate=1e-6 \
  --ema-weight-decay=0.9999 \
  --initial-weight="pretrained_models/last.pt" \
  --loss-type="CFM" \
  --use-dsm \
  --dsm-reg=5

# CC-DPO (beta=0.1)
# Best FID: 2.32646 at iteration 350.
# Best FD_DINOv2: 47.1951 at iteration 750.
accelerate launch --num_processes 1 MCLR_finetuning.py --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --encoder-depth=8 \
  --enc-type="None" \
  --output-dir="exps_CC_DPO" \
  --exp-name="cc_dpo_finetuning_repa_ema_0.9999_lr_1e-6_batchsize_64_beta_0.1" \
  --data-dir="data" \
  --batch-size=64 \
  --checkpointing-steps=50 \
  --sampling-steps=50 \
  --learning-rate=1e-6 \
  --ema-weight-decay=0.9999 \
  --initial-weight="pretrained_models/last.pt" \
  --loss-type="CC-DPO" \
  --beta=0.1 \
  --gradient-accumulation-steps=1

# CCA (beta=0.1, lambda=10)
# Best FID: 2.92997 at iteration 150.
# Best FD_DINOv2: 60.2044 at iteration 250.
accelerate launch --num_processes 1 MCLR_finetuning.py --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --encoder-depth=8 \
  --enc-type="None" \
  --output-dir="exps_CCA" \
  --exp-name="cca_finetuning_repa_ema_0.9999_lr_1e-6_batchsize_64_beta_0.1_lamb_10" \
  --data-dir="data" \
  --batch-size=64 \
  --checkpointing-steps=50 \
  --sampling-steps=50 \
  --learning-rate=1e-6 \
  --ema-weight-decay=0.9999 \
  --initial-weight="pretrained_models/last.pt" \
  --loss-type="CCA" \
  --beta=0.1 \
  --lamb=10 \
  --gradient-accumulation-steps=1

# MCLR + DSM (reg=5)
# Best FID: 2.51200 at iteration 15000.
# Best FD_DINOv2: 84.1459 at iteration 19000.
# Historical results include a resumed continuation beyond the initial 10000 steps.
accelerate launch --num_processes 1 MCLR_finetuning.py --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --encoder-depth=8 \
  --enc-type="None" \
  --output-dir="exps_MCLR" \
  --exp-name="mclr_finetuning_dsm_reg5_ema_0.9999_lr_1e-6" \
  --data-dir="data" \
  --batch-size=64 \
  --max-train-steps=20000 \
  --checkpointing-steps=100 \
  --sampling-steps=50 \
  --learning-rate=1e-6 \
  --ema-weight-decay=0.9999 \
  --initial-weight="pretrained_models/last.pt" \
  --loss-type="MCLR" \
  --use-dsm \
  --dsm-reg=5
