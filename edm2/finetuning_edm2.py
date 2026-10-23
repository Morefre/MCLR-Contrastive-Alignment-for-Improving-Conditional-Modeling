# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion models according to the EDM2 recipe from the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.finetuning_loop

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2-img512-xs':   dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-s':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=192, lr=0.0100, decay=70000, dropout=0.00, P_mean=-0.4, P_std=1.0),
    'edm2-img512-m':    dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-l':    dnnlib.EasyDict(duration=1792<<20, batch=2048, channels=320, lr=0.0080, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xl':   dnnlib.EasyDict(duration=1280<<20, batch=2048, channels=384, lr=0.0070, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img512-xxl':  dnnlib.EasyDict(duration=896<<20,  batch=2048, channels=448, lr=0.0065, decay=70000, dropout=0.10, P_mean=-0.4, P_std=1.0),
    'edm2-img64-s':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img64-m':     dnnlib.EasyDict(duration=2048<<20, batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-l':     dnnlib.EasyDict(duration=1024<<20, batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),
    'edm2-img64-xl':    dnnlib.EasyDict(duration=640<<20,  batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6),

    'edm2-img64-s-MCLR-accumulated': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.00001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=3.0, negative_pairs_per_chunk=25),
    'edm2-img64-s-MCLR-simplified': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=3.0),
    'edm2-img64-s-MCLR-simplified-DSM': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=3.0, P_mean_dsm=-0.8, P_std_dsm=1.6),
    'edm2-img512-l-MCLR-accumulated': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=320, lr=0.00001, decay=35000, dropout=0.00, P_mean=-0.4, P_std=3.0, negative_pairs_per_chunk=25),
    'edm2-img512-l-MCLR-simplified': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=320, lr=0.00001, decay=35000, dropout=0.00, P_mean=-0.4, P_std=3.0),

    'edm2-img64-s-DPO-accumulated': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=3.0, negative_pairs_per_chunk=15),
    'edm2-img64-s-DPO-simplified': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6),
    'edm2-img512-l-DPO-accumulated': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=320, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.4, P_std=3.0, negative_pairs_per_chunk=15),
    'edm2-img512-l-DPO-simplified': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=320, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.4, P_std=3.0),

    'edm2-img64-s-CCA-accumulated': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=3.0, negative_pairs_per_chunk=15),
    'edm2-img64-s-CCA-simplified': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=192, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.8, P_std=3.0),
    'edm2-img512-l-CCA-accumulated': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=320, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.4, P_std=3.0, negative_pairs_per_chunk=15),
    'edm2-img512-l-CCA-simplified': dnnlib.EasyDict(duration=1024<<20, batch=560, channels=320, lr=0.0001, decay=35000, dropout=0.00, P_mean=-0.4, P_std=3.0),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2-img512-s', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.get('cond', True))
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Hyperparameters.
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks_edm2.Precond', model_channels=opts.channels, dropout=opts.dropout)
    #c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    if preset in {'edm2-img64-s-MCLR-simplified', 'edm2-img512-l-MCLR-simplified'}:
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_simplified', P_mean=opts.P_mean, P_std=opts.P_std)
    elif preset in {'edm2-img64-s-MCLR-accumulated', 'edm2-img512-l-MCLR-accumulated'}:
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_Accumulated', P_mean=opts.P_mean, P_std=opts.P_std, negative_pairs_per_chunk=opts.negative_pairs_per_chunk)
    elif preset == 'edm2-img64-s-MCLR-simplified-DSM':
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.MRLoss_ImageNet_EDM_Sampling_EDM_Weighting_simplified_with_DSM', P_mean=opts.P_mean, P_std=opts.P_std, P_mean_dsm=opts.P_mean_dsm, P_std_dsm=opts.P_std_dsm)
    elif preset in {'edm2-img64-s-DPO-simplified', 'edm2-img512-l-DPO-simplified'}:
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.DPO_Loss_EDM_Sampling_EDM_Weighting_simplified', P_mean=opts.P_mean, P_std=opts.P_std)
    elif preset in {'edm2-img64-s-DPO-accumulated', 'edm2-img512-l-DPO-accumulated'}:
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.DPO_Loss_EDM_Sampling_EDM_Weighting_Accumulated', P_mean=opts.P_mean, P_std=opts.P_std, negative_pairs_per_chunk=opts.negative_pairs_per_chunk)
    elif preset in {'edm2-img64-s-CCA-simplified', 'edm2-img512-l-CCA-simplified'}:
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.CCA_Loss_EDM_Sampling_EDM_Weighting_simplified', P_mean=opts.P_mean, P_std=opts.P_std)
    elif preset in {'edm2-img64-s-CCA-accumulated', 'edm2-img512-l-CCA-accumulated'}:
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.finetuning_loop.CCA_Loss_EDM_Sampling_EDM_Weighting_Accumulated', P_mean=opts.P_mean, P_std=opts.P_std, negative_pairs_per_chunk=opts.negative_pairs_per_chunk)


    c.lr_kwargs = dnnlib.EasyDict(func_name='training.finetuning_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)
    c.k = opts.k
    c.return_logvar = opts.return_logvar
    c.resume_pkl = opts.resume_pkl
    c.reference_net_pkl = opts.reference_net_pkl
    c.use_decay = opts.use_decay
    c.beta = opts.beta
    c.lamb = opts.lamb
    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.finetuning_loop.finetuning_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='edm2-img512-s', show_default=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--P_mean_dsm', 'P_mean_dsm', help='Noise level mean_dsm', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std_dsm', 'P_std_dsm',   help='Noise level standard deviation_dsm', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)
@click.option('--k',                help='Number of negative pairs', metavar='INT',             type=click.IntRange(min=1), default=10)
@click.option('--beta',             help='beta for DPO KL divergence and CCA', metavar='FLOAT',         type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lamb',             help='lambda for CCA', metavar='FLOAT',                     type=click.FloatRange(min=0, min_open=True), default=None)

@click.option('--negative_pairs_per_chunk', help='Number of negative pairs per chunk', metavar='INT',  type=click.IntRange(min=1), default=10)
@click.option('--return_logvar',    help='whether use uncertanity loss scaling', metavar='BOOL',type=bool, default=True, show_default=True)
@click.option('--resume_pkl',       help='Path to the initial network', metavar='STR',          type=str, default=None, show_default=True)
@click.option('--reference_net_pkl', help='Path to the reference network', metavar='STR',       type=str, default=None, show_default=True)
@click.option('--use_decay',        help='Whether use learning rate decay', metavar='BOOL',     type=bool, default=False, show_default=True)


# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='128Mi', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

def cmdline(outdir, dry_run, **opts):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
