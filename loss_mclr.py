import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SILoss_MCLR:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, use_dsm=False, random_dsm=False, dsm_reg=1, model_kwargs=None, zs=None, use_trust_region=False, trust_region_reg=1, ref_model=None):
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs)
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde, drop_ids  = model(model_input, time_input.flatten(), return_drop_ids=True, **model_kwargs)
        y_label_correct = model_kwargs['y']
        if drop_ids is None:
            y_label_correct_with_drop_out = y_label_correct
        else:
            y_label_correct_with_drop_out = torch.where(drop_ids, 1000, y_label_correct)
        y_label_wrong_with_drop_out = torch.roll(y_label_correct_with_drop_out, shifts=-1, dims=0)
        model_kwargs_wrong = dict(model_kwargs)
        model_kwargs_wrong['y'] = y_label_wrong_with_drop_out
        model_output_wrong, _ = model(model_input, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)

        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_wrong = mean_flat((model_output_wrong - model_target) **2)

        if use_dsm == True and random_dsm == True:
            if self.weighting == "uniform":
                time_input_dsm_reg = torch.rand((images.shape[0], 1, 1, 1))
            elif self.weighting == "lognormal":
                rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
                sigma = rnd_normal.exp()
                if self.path_type == "linear":
                    time_input_dsm_reg = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input_dsm_reg = 2 / np.pi * torch.atan(sigma)

            time_input_dsm_reg = time_input_dsm_reg.to(device=images.device, dtype=images.dtype)
            noises_dsm_reg = torch.randn_like(images)
            alpha_t_dsm_reg, sigma_t_dsm_reg, d_alpha_t_dsm_reg, d_sigma_t_dsm_reg = self.interpolant(time_input_dsm_reg)

            model_input_dsm_reg = alpha_t_dsm_reg * images + sigma_t_dsm_reg * noises_dsm_reg

            if self.prediction == 'v':
                model_target_dsm_reg = d_alpha_t_dsm_reg * images + d_sigma_t_dsm_reg * noises_dsm_reg
            else:
                raise NotImplementedError() # TODO: add x or eps prediction

            model_output_dsm_reg, _ = model(
                model_input_dsm_reg,
                time_input_dsm_reg.flatten(),
                use_label_dropout=False,
                return_drop_ids=False,
                **model_kwargs,
            )
            denoising_loss_dsm_reg = mean_flat((model_output_dsm_reg - model_target_dsm_reg) ** 2)

        if use_dsm == False:
            loss = denoising_loss - denoising_loss_wrong
        elif use_dsm == True:
            if random_dsm == False:
                loss = (dsm_reg + 1) * denoising_loss - denoising_loss_wrong
            elif random_dsm == True:
                loss = dsm_reg * denoising_loss_dsm_reg + denoising_loss - denoising_loss_wrong

        if use_trust_region == True:
            assert ref_model != None
            model_kwargs_ref = dict(model_kwargs)
            model_kwargs_ref['y'] = y_label_correct_with_drop_out
            with torch.no_grad():
                model_output_ref, _ = ref_model(
                    model_input,
                    time_input.flatten(),
                    use_label_dropout=False,
                    return_drop_ids=False,
                    **model_kwargs_ref,
                )
            trust_region_loss = mean_flat((model_output_ref - model_output) ** 2)
            loss = loss + trust_region_loss * trust_region_reg

        return loss

class SILoss_CFM:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, use_dsm=False, random_dsm=False, dsm_reg=1, model_kwargs=None, zs=None, use_trust_region=False, trust_region_reg=1, ref_model=None):
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs)
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde, drop_ids  = model(model_input, time_input.flatten(), return_drop_ids=True, **model_kwargs)
        y_label_correct = model_kwargs['y']
        if drop_ids is None:
            y_label_correct_with_drop_out = y_label_correct
        else:
            y_label_correct_with_drop_out = torch.where(drop_ids, 1000, y_label_correct)
        
        # construct mismatched samples
        images_wrong = torch.roll(images, shifts=-1, dims=0)
        # use randomly sampled noise and mismatched samples to construct the wrong target
        model_target_wrong = d_alpha_t * images_wrong + d_sigma_t * torch.randn_like(images_wrong)
        
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_wrong = mean_flat((model_output - model_target_wrong) **2)

        # random dsm means the dsm term is different from that in the contrastive term, in CFM, the two are the same
        if use_dsm == True and random_dsm == True:
            if self.weighting == "uniform":
                time_input_dsm_reg = torch.rand((images.shape[0], 1, 1, 1))
            elif self.weighting == "lognormal":
                rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
                sigma = rnd_normal.exp()
                if self.path_type == "linear":
                    time_input_dsm_reg = sigma / (1 + sigma)
                elif self.path_type == "cosine":
                    time_input_dsm_reg = 2 / np.pi * torch.atan(sigma)

            time_input_dsm_reg = time_input_dsm_reg.to(device=images.device, dtype=images.dtype)
            noises_dsm_reg = torch.randn_like(images)
            alpha_t_dsm_reg, sigma_t_dsm_reg, d_alpha_t_dsm_reg, d_sigma_t_dsm_reg = self.interpolant(time_input_dsm_reg)

            model_input_dsm_reg = alpha_t_dsm_reg * images + sigma_t_dsm_reg * noises_dsm_reg

            if self.prediction == 'v':
                model_target_dsm_reg = d_alpha_t_dsm_reg * images + d_sigma_t_dsm_reg * noises_dsm_reg
            else:
                raise NotImplementedError() # TODO: add x or eps prediction

            model_output_dsm_reg, _ = model(
                model_input_dsm_reg,
                time_input_dsm_reg.flatten(),
                use_label_dropout=False,
                return_drop_ids=False,
                **model_kwargs,
            )
            denoising_loss_dsm_reg = mean_flat((model_output_dsm_reg - model_target_dsm_reg) ** 2)

        if use_dsm == False:
            loss = denoising_loss - denoising_loss_wrong
        elif use_dsm == True:
            if random_dsm == False:
                loss = (dsm_reg + 1) * denoising_loss - denoising_loss_wrong
            elif random_dsm == True:
                loss = dsm_reg * denoising_loss_dsm_reg + denoising_loss - denoising_loss_wrong

        if use_trust_region == True:
            assert ref_model != None
            model_kwargs_ref = dict(model_kwargs)
            model_kwargs_ref['y'] = y_label_correct_with_drop_out
            with torch.no_grad():
                model_output_ref, _ = ref_model(
                    model_input,
                    time_input.flatten(),
                    use_label_dropout=False,
                    return_drop_ids=False,
                    **model_kwargs_ref,
                )
            trust_region_loss = mean_flat((model_output_ref - model_output) ** 2)
            loss = loss + trust_region_loss * trust_region_reg

        return loss


class SILoss_CC_DPO:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, ref_model, images, beta=0.1, model_kwargs=None):
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs)
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde, drop_ids  = model(model_input, time_input.flatten(), return_drop_ids=True, **model_kwargs)

        y_label_correct = model_kwargs['y']
        if drop_ids is None:
            y_label_correct_with_drop_out = y_label_correct
        else:
            y_label_correct_with_drop_out = torch.where(drop_ids, 1000, y_label_correct)
        
        # here we force the negative samples be corrupted by the same sets of noise as the positive samples.
        # our preliminary results imply that mismatched noises also work well, i.e., we can also do models_input_wrong = torch.roll(model_input, shifts=-1, dims=0)
        images_wrong = torch.roll(images, shifts=-1, dims=0)
        model_input_wrong = alpha_t * images_wrong + sigma_t * noises
        if self.prediction == 'v':
            model_target_wrong = d_alpha_t * images_wrong + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        
        model_kwargs_wrong = dict(model_kwargs)
        model_kwargs_wrong['y'] = y_label_correct_with_drop_out
        model_output_wrong, _ = model(model_input_wrong, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)

        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_wrong = mean_flat((model_output_wrong - model_target_wrong) **2)

        with torch.no_grad():
            model_output_ref, _ = ref_model(model_input, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)
            model_output_ref_wrong, _ = ref_model(model_input_wrong, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)
            denoising_loss_ref = mean_flat((model_output_ref - model_target) ** 2)
            denoising_loss_ref_wrong = mean_flat((model_output_ref_wrong - model_target_wrong) ** 2)
        
        winning_diff = denoising_loss - denoising_loss_ref
        losing_diff = denoising_loss_wrong - denoising_loss_ref_wrong

        loss = -torch.nn.functional.logsigmoid(-1*beta*(winning_diff - losing_diff)) # DPO loss
        
        return loss

class SILoss_CCA:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, ref_model, images, beta=0.1, lamb=10, model_kwargs=None):
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = dict(model_kwargs)
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde, drop_ids  = model(model_input, time_input.flatten(), return_drop_ids=True, **model_kwargs)

        y_label_correct = model_kwargs['y']
        if drop_ids is None:
            y_label_correct_with_drop_out = y_label_correct
        else:
            y_label_correct_with_drop_out = torch.where(drop_ids, 1000, y_label_correct)
        
        # here we force the negative samples be corrupted by the same sets of noise as the positive samples.
        # our preliminary results imply that mismatched noises also work well, i.e., we can also do models_input_wrong = torch.roll(model_input, shifts=-1, dims=0)
        images_wrong = torch.roll(images, shifts=-1, dims=0)
        model_input_wrong = alpha_t * images_wrong + sigma_t * noises
        if self.prediction == 'v':
            model_target_wrong = d_alpha_t * images_wrong + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        
        model_kwargs_wrong = dict(model_kwargs)
        model_kwargs_wrong['y'] = y_label_correct_with_drop_out
        model_output_wrong, _ = model(model_input_wrong, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)

        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss_wrong = mean_flat((model_output_wrong - model_target_wrong) **2)

        with torch.no_grad():
            model_output_ref, _ = ref_model(model_input, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)
            model_output_ref_wrong, _ = ref_model(model_input_wrong, time_input.flatten(), **model_kwargs_wrong, return_drop_ids=False, use_label_dropout=False)
            denoising_loss_ref = mean_flat((model_output_ref - model_target) ** 2)
            denoising_loss_ref_wrong = mean_flat((model_output_ref_wrong - model_target_wrong) ** 2)
        
        winning_diff = denoising_loss - denoising_loss_ref
        losing_diff = denoising_loss_wrong - denoising_loss_ref_wrong

        loss = -1 * (torch.nn.functional.logsigmoid(-1*beta*winning_diff) + lamb*torch.nn.functional.logsigmoid(beta*losing_diff)) # CCA loss

        return loss
