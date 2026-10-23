# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate evaluation metrics (FID and FD_DINOv2)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset
# import generate_images

#----------------------------------------------------------------------------
# Abstract base class for feature detectors.

class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x): # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# InceptionV3 feature detector.
# This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

class InceptionV3Detector(Detector):
    def __init__(self):
        super().__init__(feature_dim=2048)
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

    def __call__(self, x, return_logits=False):
        if return_logits == False:
            #print('ha')
            return self.model.to(x.device)(x, return_features=True)
        elif return_logits == True:
            feature = self.model.to(x.device)(x, no_output_bias=True)
            #print(torch.sum(feature, dim=1))
            return feature

#----------------------------------------------------------------------------
# DINOv2 feature detector.
# Modeled after https://github.com/layer6ai-labs/dgm-eval

class DINOv2Detector(Detector):
    def __init__(self, resize_mode='torch'):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if self.resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif self.resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)

#----------------------------------------------------------------------------
# Metric specifications.

metric_specs = {
    'fid':             dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'fd_dinov2':       dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector)),
    'vendi_inception': dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'vendi_dinov2':    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector)),
    'inception'   :    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'dinov2'      :    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector))
}

#----------------------------------------------------------------------------
# Get feature detector for the given metric.

_detector_cache = dict()

def get_detector(metric, verbose=True):
    # Lookup from cache.
    if metric in _detector_cache:
        return _detector_cache[metric]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        #print(f'Rank {dist.get_rank()} is waiting at the first barrier')
        torch.distributed.barrier()
        #print(f'Rank {dist.get_rank()} passed the first barrier')


    # Construct detector.
    kwargs = metric_specs[metric].detector_kwargs
    if verbose:
        name = kwargs.class_name.split('.')[-1] if isinstance(kwargs.class_name, str) else kwargs.class_name.__name__
        dist.print0(f'Setting up {name}...')
    detector = dnnlib.util.construct_class_by_name(**kwargs)
    _detector_cache[metric] = detector

    # Other ranks follow.
    if dist.get_rank() == 0:
        #print(f'Rank {dist.get_rank()} is waiting at the first barrier')
        torch.distributed.barrier()
        #print(f'Rank {dist.get_rank()} passed the first barrier')

    return detector

#----------------------------------------------------------------------------
# Load feature statistics from the given .pkl or .npz file.

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Save feature statistics to the given .pkl file.

def save_stats(stats, path, verbose=True):
    if verbose:
        print(f'Saving feature statistics to {path} ...')
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(stats, f)

#----------------------------------------------------------------------------
# Manifold Estimator for Computing the Precision and Recall.
class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57 
                 and https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py
    """

    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        self.distance_block = DistanceBlock()
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def warmup(self):
        feats = torch.zeros((1, 2048), dtype=torch.float32)
        radii = torch.zeros((1, 1), dtype=torch.float32)
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: torch.Tensor) -> torch.Tensor:
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = torch.zeros([num_images, self.num_nhoods], dtype=torch.float32).to(features.device)
        distance_batch = torch.zeros([self.row_batch_size, num_images], dtype=torch.float32)
        seq = torch.arange(max(self.nhood_sizes) + 1, dtype=torch.int32) # plus 1, so that the data itself is not counted

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            x, _ = torch.topk(distance_batch[0 : end1 - begin1, :], max(self.nhood_sizes)+1, dim=1, largest=False)
            #print(x.shape, radii.shape)
            radii[begin1:end1, :] = x[:, self.nhood_sizes]
            # radii[begin1:end1, :] = torch.cat(
            #     [
            #         x[:, self.nhood_sizes]
            #         for x in torch.topk(distance_batch[0 : end1 - begin1, :], self.num_nhoods, dim=1, largest=False)[0]
            #     ],
            #     dim=0,
            # )

        if self.clamp_to_percentile is not None:
            max_distances = torch.quantile(radii, self.clamp_to_percentile / 100.0, dim=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(self, features: torch.Tensor, radii: torch.tensor, eval_features: torch.Tensor):
        """
        Evaluate if new feature vectors are at the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = torch.zeros([self.row_batch_size, num_ref_images], dtype=torch.float32)
        batch_predictions = torch.zeros([num_eval_images, self.num_nhoods], dtype=torch.int32)
        max_realism_score = torch.zeros([num_eval_images], dtype=torch.float32)
        nearest_indices = torch.zeros([num_eval_images], dtype=torch.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= radii
            batch_predictions[begin1:end1] = torch.any(samples_in_manifold, dim=1).int()

            max_realism_score[begin1:end1] = torch.max(
                radii[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), dim=1
            ).values
            nearest_indices[begin1:end1] = torch.argmin(distance_batch[0 : end1 - begin1, :], dim=1)

        return {
            "fraction": float(batch_predictions.float().mean().item()),
            "batch_predictions": batch_predictions,
            "max_realism_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: torch.Tensor,
        radii_1: torch.Tensor,
        features_2: torch.Tensor,
        radii_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N2 x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: a tensor of length K1
                 - recall: an tensor of length K2
        """
        features_1_status = torch.zeros([len(features_1), radii_2.shape[1]], dtype=torch.bool).to(features_1.device)
        features_2_status = torch.zeros([len(features_2), radii_1.shape[1]], dtype=torch.bool).to(features_2.device)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in

        return (
            features_2_status.float().mean(dim=0),
            features_1_status.float().mean(dim=0),
        )


class DistanceBlock:
    """
    Calculate pairwise distances between vectors in PyTorch.
    
    Adapted from the TensorFlow implementation.
    """

    def __init__(self):
        """
        Initializes the class but does not require session handling as in TensorFlow.
        """
        pass

    def _batch_pairwise_distances(self, U, V, dtype=torch.float16):
        """
        Compute pairwise squared Euclidean distances between two batches of feature vectors.
        Uses float16 for efficiency but falls back to float32 if necessary.
        """
        U = U.to(dtype).reshape(U.shape[0], -1)  # Flatten input
        V = V.to(dtype).reshape(V.shape[0], -1)  # Flatten input

        norm_u = torch.sum(U ** 2, dim=1, keepdim=True)  # (B, 1)
        norm_v = torch.sum(V ** 2, dim=1, keepdim=True)  # (B, 1)

        D = norm_u - 2 * (U @ V.T) + norm_v.T  # Compute squared distances
        D = torch.clamp(D, min=0.0)  # Ensure non-negative distances

        # Check if the computed distance matrix has NaN or Inf values
        if torch.all(torch.isfinite(D)):
            return D.to(torch.float32)  # Convert to float32 for final output
        else:
            return self._batch_pairwise_distances(U.to(torch.float32), V.to(torch.float32), dtype=torch.float32)

    def pairwise_distances(self, U, V):
        """
        Compute pairwise distances between two batches of feature vectors.
        """
        return self._batch_pairwise_distances(U, V)

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        """
        Compute whether each sample in batch_1 has a distance within radii_2 of any sample in batch_2.
        Similarly, compute for batch_2 within radii_1.
        """
        D = self._batch_pairwise_distances(batch_1, batch_2)  # Compute distance matrix

        # Expand dimensions for broadcasting
        dist_expanded = D.unsqueeze(-1)  # Shape (B1, B2, 1)

        # Compare with radii and get boolean masks
        batch_1_in = torch.any(dist_expanded <= radii_2, dim=1)  # Shape (B1,)
        batch_2_in = torch.any(dist_expanded <= radii_1.unsqueeze(1), dim=0)  # Shape (B2,)

        return batch_1_in, batch_2_in


#------------------------------------------------------------------------------------------------
# Get the inception and dino feature embeddings for efficient calculation of Precision and Recall
def get_features(
    image_path, # Path to a directory or ZIP file containing the images.        
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0, # Random seed for selecting images.
    max_batch_size  = 64, # Maximum batch size.
    num_workers     = 2, # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    metrics         = ['inception', 'dinov2'],
    device          = torch.device('cuda'),
    save_path       = None, # path for saving the features
    ):

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))
    
    num_batches = len(data_loader)
    detectors = []
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    for i in range(len(metrics)):
        # Set the detector
        metric = metrics[i]
        detector = detectors[i]

        # Set the array for storing the features
        features_all = []

        # Loop over sample batches.
        for batch_idx, images in enumerate(data_loader):
            if isinstance(images, dict) and 'images' in images: # dict(images)
                images = images['images']
            elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                images = images[0]
            images = torch.as_tensor(images).to(device)

            # Accumulate statistics.
            if images is not None:
                features = detector(images).to(torch.float32)
                features_all.append(features.reshape(features.shape[0],-1))
            
        features_all = torch.cat(features_all)
        features_gathered_sample = [torch.empty_like(features_all) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(features_gathered_sample, features_all)
        features_all = torch.cat(features_gathered_sample, dim=0)

        if save_path != None and dist.get_rank()==0:
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            if metric == 'inception':
                torch.save(features_all, os.path.join(save_path, 'inception_features.pt'))
            elif metric == 'dinov2':
                torch.save(features_all, os.path.join(save_path, 'dinov2_features.pt'))
    
    torch.distributed.barrier()

def calc_inception_score(
    image_path,
    verbose = True,
    num_splits = 10,
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0, # Random seed for selecting images.
    max_batch_size  = 64, # Maximum batch size.
    num_workers     = 2, # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    device = torch.device('cuda'),
    save_path = None
    ):

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))    

    num_batches = len(data_loader)
    detector = get_detector('inception')
    if verbose:
        dist.print0('Calculating feature statistics...')
    
    # Set the array for storing the features
    preds = []

    # Loop over sample batches.
    for batch_idx, images in enumerate(data_loader):
        if isinstance(images, dict) and 'images' in images: # dict(images)
            images = images['images']
        elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
            images = images[0]
        images = torch.as_tensor(images).to(device)

        # Accumulate statistics.
        if images is not None:
            features = detector(images, return_logits=True).to(torch.float32)
            preds.append(features.reshape(features.shape[0],-1))

    preds = torch.cat(preds)
    preds_gathered = [torch.empty_like(preds) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(preds_gathered, preds)
    preds = torch.cat(preds_gathered, dim=0)
    
    #print(preds.shape)

    if dist.get_rank() == 0:
        num_gen = len(dataset_obj)
        #print(num_gen)
        #print(preds.shape[0])
        scores = []
        for i in range(num_splits):
            part = preds[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
            kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0, keepdim=True)))
            kl = torch.mean(torch.sum(kl, dim=1))
            scores.append(torch.exp(kl))
        is_score = torch.mean(torch.tensor(scores)).item()

        if verbose == True:
            print('IS Score:',is_score)
        
        if save_path != None:
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            torch.save(is_score, os.path.join(save_path,'inception_score.pt'))


def calc_prec_recall_for_iterable(
    image_iter,                                        # Iterable of image batches: NCHW, uint8, 3 channels.
    ref_iter=None,                                          # Iterable of reference image batches: NCHW, uint8, 3 channels.
    image_feature_path = None,
    ref_feature_path =None,    
    metrics         = ['inception', 'dinov2'],         # Metrics to compute the statistics for.
    verbose         = True,                            # Metrics to compute the statistics for.
    row_batch_size  = 10000,
    col_batch_size=10000,
    nhood_sizes=(3,),
    clamp_to_percentile=None,
    eps=1e-5,
    device          = torch.device('cuda'),
    save_path       = None,                            # Path for saving the vendi scores
    ):
        # Return an interable for calculating the statistics.
    num_batches = len(image_iter)
    detectors = []
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Initialize a dictionary for storing perception and recall
    state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]

    for s in state:
        s.precision = -1
        s.recall = -1

    for i in range(len(metrics)):
        # Set the detector
        metric = metrics[i]
        for s in state:
            if s.metric == metric:
                detector = s.detector
                current_s = s

        if image_feature_path != None:
            if metric == 'inception':
                features_all_sample_batch = torch.load(os.path.join(image_feature_path, 'inception_features.pt'), map_location='cpu')
            elif metric == 'dinov2':
                features_all_sample_batch = torch.load(os.path.join(image_feature_path, 'dinov2_features.pt'), map_location='cpu')
            features_all_sample_batch = features_all_sample_batch.to(device)
        else:
            # Set the array for storing the features
            features_all_sample_batch = []

            # Loop over sample batches.
            for batch_idx, images in enumerate(image_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    images = images['images']
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    images = images[0]
                images = torch.as_tensor(images).to(device)

                # Accumulate statistics.
                if images is not None:
                    features = detector(images).to(torch.float32)
                    features_all_sample_batch.append(features.reshape(features.shape[0],-1))

            features_all_sample_batch = torch.cat(features_all_sample_batch)
            features_gathered_sample = [torch.empty_like(features_all_sample_batch) for _ in range(dist.get_world_size())]
            torch.distributed.all_gather(features_gathered_sample, features_all_sample_batch)
            features_all_sample_batch = torch.cat(features_gathered_sample, dim=0)

        if ref_feature_path != None:
            if metric == 'inception':
                features_all_reference_batch = torch.load(os.path.join(ref_feature_path, 'inception_features.pt'), map_location='cpu')
            elif metric == 'dinov2':
                features_all_reference_batch = torch.load(os.path.join(ref_feature_path, 'dinov2_features.pt'), map_location='cpu')
            features_all_reference_batch = features_all_reference_batch.to(device)
        else:
            # Set the array for storing the features
            features_all_reference_batch = []

            # Loop over reference batches.
            for batch_idx, images in enumerate(ref_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    images = images['images']
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    images = images[0]
                images = torch.as_tensor(images).to(device)

                # Accumulate statistics.
                if images is not None:
                    features = detector(images).to(torch.float32)
                    features_all_reference_batch.append(features.reshape(features.shape[0],-1))

            features_all_reference_batch = torch.cat(features_all_reference_batch)
            features_gathered_ref = [torch.empty_like(features_all_reference_batch) for _ in range(dist.get_world_size())]
            torch.distributed.all_gather(features_gathered_ref, features_all_reference_batch)
            features_all_reference_batch = torch.cat(features_gathered_ref, dim=0)


        if dist.get_rank() == 0:
            # Create the manifold estimator
            manifold_estimator = ManifoldEstimator(row_batch_size=row_batch_size, col_batch_size=col_batch_size, nhood_sizes=nhood_sizes, clamp_to_percentile=clamp_to_percentile, eps=eps)
            radii_1 = manifold_estimator.manifold_radii(features_all_reference_batch)
            radii_2 = manifold_estimator.manifold_radii(features_all_sample_batch)
            pr = manifold_estimator.evaluate_pr(
                features_all_reference_batch, radii_1, features_all_sample_batch, radii_2
            )
            prec, recall = float(pr[0][0]), float(pr[1][0])
            current_s.precision = prec
            current_s.recall = recall 
    
    if verbose == True and  dist.get_rank() ==0:
        for s in state:
            print('metrics:',s.metric)
            print('precision:',s.precision)
            print('recall:',s.recall)

    if save_path != None and dist.get_rank()==0:
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        state_filtered = [{k: v for k, v in s.items() if k in ['metric', 'precision', 'recall']} for s in state]
        torch.save(state_filtered, os.path.join(save_path,'prec_and_recall.pt'))

    # if save_path != None and dist.get_rank()==0:
    #     if os.path.exists(save_path) == False:
    #         os.makedirs(save_path)
    #     #state_filtered = [{k: v for k, v in s.items() if k in ['metric', 'score', 'S']} for s in state]
    #     torch.save(state, os.path.join(save_path,'prec_and_recall.pt'))

#----------------------------------------------------------------------------
# Calculate Precision and Recall
def calc_prec_recall(
    image_path, # Path to a directory or ZIP file containing the images.
    ref_path = None,   # Path to the reference images    
    image_feature_path = None,
    ref_feature_path =None,    
    row_batch_size=10000,
    col_batch_size=10000,
    nhood_sizes=(3,),
    clamp_to_percentile=None,
    eps=1e-5,
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0, # Random seed for selecting images.
    max_batch_size  = 64, # Maximum batch size.
    num_workers     = 2, # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    metrics         = ['inception', 'dinov2'],
    device          = torch.device('cuda'),
    save_path       = None, # path for saving the vendi scores
    ):

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    if ref_path == None:
        calc_prec_recall_for_iterable(image_iter=data_loader, image_feature_path=image_feature_path, ref_feature_path=ref_feature_path, metrics=metrics, verbose=verbose, device=device, save_path=save_path, row_batch_size=row_batch_size, col_batch_size=col_batch_size, nhood_sizes=nhood_sizes, clamp_to_percentile=clamp_to_percentile, eps=eps)
        torch.distributed.barrier()

    else:
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        # Load reference images.
        if verbose:
            dist.print0(f'Loading images from {ref_path} ...')

        dataset_obj_ref = dataset.ImageFolderDataset(path=ref_path, max_size=num_images, random_seed=seed)
        if num_images is not None and len(dataset_obj_ref) < num_images:
            raise click.ClickException(f'Found {len(dataset_obj_ref)} images, but expected at least {num_images}')
        if len(dataset_obj) < 2:
            raise click.ClickException(f'Found {len(dataset_obj_ref)} images, but need at least 2 to compute statistics')

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        # Divide images into batches.
        num_batches = max((len(dataset_obj_ref) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
        rank_batches = np.array_split(np.arange(len(dataset_obj_ref)), num_batches)[dist.get_rank() :: dist.get_world_size()]
        data_loader_ref = torch.utils.data.DataLoader(dataset_obj_ref, batch_sampler=rank_batches,
            num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))
        
        calc_prec_recall_for_iterable(image_iter=data_loader, ref_iter=data_loader_ref, image_feature_path=image_feature_path, ref_feature_path=ref_feature_path, metrics=metrics, verbose=verbose, device=device, save_path=save_path, row_batch_size=row_batch_size, col_batch_size=col_batch_size, nhood_sizes=nhood_sizes, clamp_to_percentile=clamp_to_percentile, eps=eps)
        torch.distributed.barrier()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                         # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics     = ['fid', 'fd_dinov2'], # Metrics to compute the statistics for.
    verbose     = True,                 # Enable status prints?
    dest_path   = None,                 # Where to save the statistics. None = do not save.
    device      = torch.device('cuda'), # Which compute device to use.
):
    # Initialize.
    num_batches = len(image_iter)
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Convenience wrapper for torch.distributed.all_reduce().
    def all_reduce(x):
        x = x.clone()
        torch.distributed.all_reduce(x)
        return x

    # Return an iterable over the batches.
    class StatsIterable:
        def __len__(self):
            return num_batches

        def __iter__(self):
            state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]
            for s in state:
                s.cum_mu = torch.zeros([s.detector.feature_dim], dtype=torch.float64, device=device)
                s.cum_sigma = torch.zeros([s.detector.feature_dim, s.detector.feature_dim], dtype=torch.float64, device=device)
            cum_images = torch.zeros([], dtype=torch.int64, device=device)

            # Loop over batches.
            for batch_idx, images in enumerate(image_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    images = images['images']
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    images = images[0]
                images = torch.as_tensor(images).to(device)

                # Accumulate statistics.
                if images is not None:
                    for s in state:
                        features = s.detector(images).to(torch.float64)
                        s.cum_mu += features.sum(0)
                        s.cum_sigma += features.T @ features
                    cum_images += images.shape[0]

                # Output results.
                r = dnnlib.EasyDict(stats=None, images=images, batch_idx=batch_idx, num_batches=num_batches)
                r.num_images = int(all_reduce(cum_images).cpu())
                if batch_idx == num_batches - 1:
                    assert r.num_images >= 2
                    r.stats = dict(num_images=r.num_images)
                    for s in state:
                        mu = all_reduce(s.cum_mu) / r.num_images
                        sigma = (all_reduce(s.cum_sigma) - mu.ger(mu) * r.num_images) / (r.num_images - 1)
                        r.stats[s.metric] = dict(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
                    if dest_path is not None and dist.get_rank() == 0:
                        save_stats(stats=r.stats, path=dest_path, verbose=False)
                yield r

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_files(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 2,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)


def calc_vendi_for_iterable(
    image_iter,                                        # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics         = ['vendi_inception', 'vendi_dinov2'], # Metrics to compute the statistics for.
    verbose         = True,                                # Metrics to compute the statistics for.
    calculate_pixel = True,                            # Set to True if computing vendi score directly in pixel space.
    rank            = 2000,                            # Specify rank for computing svd of the correlation matrices (for efficient computing purposes).
    device          = torch.device('cuda'),
    save_path       = None,                            # Path for saving the vendi scores
    ):
        # Return an interable for calculating the statistics.
    num_batches = len(image_iter)
    detectors = []
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Initialize a dictionary for storing vendi scores
    state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]
    if calculate_pixel == True:
        state.append(dnnlib.EasyDict(metric='vendi_pixel'))
    for s in state:
        s.score = 0
        s.S = None

    for i in range(len(metrics)):
        # Set the detector
        metric = metrics[i]
        for s in state:
            if s.metric == metric:
                detector = s.detector
                current_s = s
        # Set the array for storing the features
        features_all = []

        # Loop over batches.
        for batch_idx, images in enumerate(image_iter):
            if isinstance(images, dict) and 'images' in images: # dict(images)
                images = images['images']
            elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                images = images[0]
            images = torch.as_tensor(images).to(device)

            # Accumulate statistics.
            if images is not None:
                features = detector(images).to(torch.float32)
                features_all.append(features.reshape(features.shape[0],-1))
            
        features_all = torch.cat(features_all)
        features_gathered = [torch.empty_like(features_all) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(features_gathered, features_all)
        features_all = torch.cat(features_gathered, dim=0)
        #print(features_all.shape)

        if dist.get_rank() == 0:
            # Normalize features
            features_all = features_all/(torch.linalg.norm(features_all,dim=1,keepdim=True)+1e-8)
            # Calculate Correlation matrix
            corr_matrix = features_all@features_all.T/features_all.shape[0]
            _, S, _ = torch.svd_lowrank(corr_matrix, q=rank)
            S_stable = torch.clone(S)
            S_stable[S_stable==0] = 1e-8 
            entropy = -torch.sum(S*torch.log(S_stable))
            vendi_score = torch.exp(entropy)
            current_s.score = vendi_score
            current_s.S = S.detach().cpu() 

    # Calculate Pixel level vendi score
    if calculate_pixel == True:
        for s in state:
            if s.metric == 'vendi_pixel':
                current_s = s
        features_all = []

        # Loop over batches.
        for batch_idx, images in enumerate(image_iter):
            if isinstance(images, dict) and 'images' in images: # dict(images)
                images = images['images']
            elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                images = images[0]
            images = torch.as_tensor(images).to(device)

            # Accumulate statistics.
            if images is not None:
                features = images.to(torch.float32)
                features_all.append(features.reshape(features.shape[0],-1))
            
        features_all = torch.cat(features_all)

        features_gathered = [torch.empty_like(features_all) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(features_gathered, features_all)
        features_all = torch.cat(features_gathered, dim=0)
        #print(features_all.shape)

        if dist.get_rank() == 0:
            # Normalize features
            features_all = features_all/(torch.linalg.norm(features_all,dim=1,keepdim=True)+1e-8)
            # Calculate Correlation matrix
            corr_matrix = features_all@features_all.T/features_all.shape[0]
            _, S, _ = torch.svd_lowrank(corr_matrix, q=rank)
            S_stable = torch.clone(S)
            S_stable[S_stable==0] = 1e-8 
            entropy = -torch.sum(S*torch.log(S_stable))
            vendi_score = torch.exp(entropy)
            current_s.score = vendi_score
            current_s.S = S.detach().cpu() 

    
    if verbose == True and  dist.get_rank() ==0:
        for s in state:
            print('metrics:',s.metric)
            print('score:',s.score)

    if save_path != None and dist.get_rank()==0:
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        state_filtered = [{k: v for k, v in s.items() if k in ['metric', 'score', 'S']} for s in state]
        torch.save(state_filtered, os.path.join(save_path,'vendi_score.pt'))




#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)
def calc_vendi_score(
    image_path, # Path to a directory or ZIP file containing the images.                 
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0, # Random seed for selecting images.
    max_batch_size  = 64, # Maximum batch size.
    num_workers     = 2, # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    metrics         = ['vendi_inception', 'vendi_dinov2'],
    calculate_pixel = True, # set to true if computing vendi score directly in pixel space.
    rank = 2000, # specify rank for computing svd of the correlation matrices (for efficient computing purposes).
    device          = torch.device('cuda'),
    save_path       = None, # path for saving the vendi scores
    ):

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    calc_vendi_for_iterable(image_iter=data_loader,metrics=metrics,verbose=verbose, calculate_pixel=calculate_pixel, rank=rank, device=device, save_path=save_path)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# Calculate Inception or Dinov2 embeddings for T-SNE visualization
def get_embeddings(
        image_path,              # Path to a directory or ZIP file containing the images.
        num_images      = None,  # Number of images to use. None = all available images.
        seed            = 0,     # Random seed for selecting images.
        max_batch_size  = 64, # Maximum batch size.
        num_workers     = 2, # How many subprocesses to use for data loading.
        prefetch_factor = 2,    # Number of images loaded in advance by each worker.
        verbose         = True, # Enable status prints?
        metrics         = ['inception', 'dinov2'],
        device          = torch.device('cuda'),
        save_path       = None, # path for saving the embeddings
):
        # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    detectors = []
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]

    if verbose:
        dist.print0('Calculating feature statistics...')

    # if os.path.exists(save_path) == False:
    #     os.makedirs(save_path)
    
    for i in range(len(detectors)):
        detector = detectors[i]
        features_all_batch = []

        # Loop over sample batches.
        for batch_idx, images in enumerate(data_loader):
            if isinstance(images, dict) and 'images' in images: # dict(images)
                images = images['images']
            elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                images = images[0]
            images = torch.as_tensor(images).to(device)

            # Accumulate statistics.
            if images is not None:
                features = detector(images).to(torch.float32)
                features_all_batch.append(features.reshape(features.shape[0],-1))

        features_all_batch = torch.cat(features_all_batch)
        features_gathered_sample = [torch.empty_like(features_all_batch) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(features_gathered_sample, features_all_batch)
        features_all_batch = torch.cat(features_gathered_sample, dim=0)            
        features_all_batch = features_all_batch.detach().cpu().numpy()
        
        if save_path != None and dist.get_rank()==0:
            if i==0:
                save_path_metric = os.path.join(save_path, 'Inception')
            elif i==1:
                save_path_metric = os.path.join(save_path, 'DinoV2')

            if os.path.exists(save_path_metric) == False:
                os.makedirs(save_path_metric)
            
            np.save(os.path.join(save_path_metric,'embeddings.npy'), features_all_batch)

    torch.distributed.barrier()

#----------------------------------------------------------------------------
# Calculate metrics based on the given feature statistics.

def calculate_metrics_from_stats(
    stats,                          # Feature statistics of the generated images.
    ref,                            # Reference statistics of the dataset. Can be a path or URL.
    metrics = ['fid', 'fd_dinov2'], # List of metrics to compute.
    verbose = True,                 # Enable status prints?
):
    if isinstance(ref, str):
        ref = load_stats(ref, verbose=verbose)
    results = dict()
    for metric in metrics:
        if metric not in stats or metric not in ref:
            if verbose:
                print(f'No statistics computed for {metric} -- skipping.')
            continue
        if verbose:
            print(f'Calculating {metric}...')
        m = np.square(stats[metric]['mu'] - ref[metric]['mu']).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(stats[metric]['sigma'], ref[metric]['sigma']), disp=False)
        value = float(np.real(m + np.trace(stats[metric]['sigma'] + ref[metric]['sigma'] - s * 2)))
        results[metric] = value
        if verbose:
            print(f'{metric} = {value:g}')
    return results

#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """Calculate evaluation metrics (FID and FD_DINOv2).

    Examples:

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img512-xxl-guid-fid --outdir=out --subdirs --seeds=0-49999

    \b
    # Calculate metrics for a random subset of 50000 images in out/
    python calculate_metrics.py calc --images=out \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl

    \b
    # Calculate metrics directly for a given model without saving any images
    torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl \\
        --seed=123456789

    \b
    # Compute dataset reference statistics
    python calculate_metrics.py ref --data=datasets/my-dataset.zip \\
        --dest=fid-refs/my-dataset.pkl
    """
#----------------------------------------------------------------------------
# 'calc_inception' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the sample images', metavar='PATH|ZIP',                                type=str, required=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                                       type=click.IntRange(min=2), default=None, show_default=True)
@click.option('--save_path',                help='path for saving the vendi scores', metavar='PATH|ZIP',                         type=str, required=False, default=None)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                                            type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',                          type=click.IntRange(min=0), default=2, show_default=True)

def calc_inception(**opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    #if dist.get_rank() == 0:
    calc_inception_score(**opts)
    torch.distributed.barrier()


#----------------------------------------------------------------------------
# 'calc_features' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the sample images', metavar='PATH|ZIP',                                type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',                                   type=parse_metric_list, default='inception,dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                                       type=click.IntRange(min=2), default=None, show_default=True)
@click.option('--save_path',                help='path for saving the vendi scores', metavar='PATH|ZIP',                         type=str, required=False, default=None)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                                            type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',                          type=click.IntRange(min=0), default=2, show_default=True)

def calc_features(**opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    #if dist.get_rank() == 0:
    get_features(**opts)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'calc_embeddings' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the sample images', metavar='PATH|ZIP',                                type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',                                   type=parse_metric_list, default='inception,dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                                       type=click.IntRange(min=2), default=None, show_default=True)
@click.option('--save_path',                help='path for saving the vendi scores', metavar='PATH|ZIP',                         type=str, required=False, default=None)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                                            type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',                          type=click.IntRange(min=0), default=2, show_default=True)

def calc_embeddings(**opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    #if dist.get_rank() == 0:
    get_embeddings(**opts)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'calc_prec_recall' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the sample images', metavar='PATH|ZIP',                                type=str, required=True)
@click.option('--ref_images', 'ref_path',   help='Path to the reference images', metavar='PATH|ZIP',                             type=str, required=False)
@click.option('--ref_feature_path',         help='Path to the reference features', metavar='PATH|ZIP',                           type=str, required=False)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',                                   type=parse_metric_list, default='inception,dinov2', show_default=True)
@click.option('--row_batch_size',           help='row batch size for computing pair-wise distainces', metavar='INT',             type=int, default=10000, show_default=True)
@click.option('--col_batch_size',           help='col batch size for computing pair-wise distainces', metavar='INT',             type=int, default=10000, show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                                       type=click.IntRange(min=2), default=None, show_default=True)
@click.option('--save_path',                help='path for saving the vendi scores', metavar='PATH|ZIP',                         type=str, required=False, default=None)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                                            type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',                          type=click.IntRange(min=0), default=2, show_default=True)

def calc_precision_recall(**opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    #if dist.get_rank() == 0:
    calc_prec_recall(**opts)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'calc_vendi' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                                       type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',                                   type=parse_metric_list, default='vendi_inception,vendi_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                                       type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--calculate_pixel',          help='whether calculate vendi score in pixel space', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--rank',                     help='rank for performing low-rank SVD', metavar='INT',                              type=click.IntRange(min=2000), default=2000, show_default=True)
@click.option('--save_path',                help='path for saving the vendi scores', metavar='PATH|ZIP',                         type=str, required=False, default=None)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc_vendi(**opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    #if dist.get_rank() == 0:
    calc_vendi_score(**opts)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'cal_call' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc_all(ref_path, metrics, image_path, **opts):
    """Calculate metrics for a given guidance method and a given class."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    
    folders = [os.path.join(image_path, folder) for folder in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, folder))]
    sorted_folder_list = sorted(folders, key=lambda x: int(x.split('_')[-1]))
    
    FD_Dino_list = []
    FID_list = []
    
    for i in range(len(sorted_folder_list)):
        path = sorted_folder_list[i]
        stats_iter = calculate_stats_for_files(metrics=metrics, image_path=path, **opts )    
        for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
            pass
        if dist.get_rank() == 0:
            results = calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
            FD_Dino_list.append(results['fd_dinov2'])
            FID_list.append(results['fid'])
        torch.distributed.barrier()
    
    torch.save(FD_Dino_list, os.path.join(image_path, 'FD_Dino_list.pt'))
    torch.save(FID_list, os.path.join(image_path, 'FID_list.pt'))


#----------------------------------------------------------------------------
# 'calc' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)
@click.option('--save_path',                help='path for saving the vendi scores', metavar='PATH|ZIP',    type=str, required=False, default=None)

def calc(ref_path, metrics, save_path, **opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        results = calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
        if save_path != None:
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            torch.save(results, os.path.join(save_path, 'FD_scores.pt'))
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'gen' subcommand.

@cmdline.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',             type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to generate', metavar='INT',             type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for the first image', metavar='INT',          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=32, show_default=True)

def gen(net, ref_path, metrics, num_images, seed, **opts):
    """Calculate metrics for a given model using default sampler settings."""
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    image_iter = generate_images.generate_images(net=net, seeds=range(seed, seed + num_images), **opts)
    stats_iter = calculate_stats_for_iterable(image_iter, metrics=metrics)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'ref' subcommand.

@cmdline.command()
@click.option('--data', 'image_path',       help='Path to the dataset', metavar='PATH|ZIP',             type=str, required=True)
@click.option('--dest', 'dest_path',        help='Destination file', metavar='PKL',                     type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',          type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT', type=click.IntRange(min=0), default=2, show_default=True)

def ref(**opts):
    """Calculate dataset reference statistics for 'calc' and 'gen'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    stats_iter = calculate_stats_for_files(**opts)
    for _r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
