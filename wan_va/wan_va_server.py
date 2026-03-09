# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


def _vram_mb():
    return torch.cuda.memory_allocated() / 1024**2

def _param_count(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def _tensor_stats(t, name="tensor"):
    return (f"{name}: shape={list(t.shape)}, dtype={t.dtype}, "
            f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
            f"mean={t.float().mean().item():.4f}, std={t.float().std().item():.4f}")


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")

        logger.info(f"[INIT] device={self.device}, dtype={self.dtype}")
        logger.info(f"[INIT] model_path={job_config.wan22_pretrained_model_name_or_path}")
        logger.info(f"[INIT] VRAM before loading: {_vram_mb():.1f} MB")
        t_start_init = time.time()

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        t0 = time.time()
        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        logger.info(f"[INIT] VAE loaded: {_param_count(self.vae):.1f}M params, "
                     f"VRAM={_vram_mb():.1f} MB, time={time.time()-t0:.1f}s")

        t0 = time.time()
        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )
        logger.info(f"[INIT] Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}, time={time.time()-t0:.1f}s")

        t0 = time.time()
        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        logger.info(f"[INIT] T5 encoder loaded: {_param_count(self.text_encoder):.1f}M params, "
                     f"VRAM={_vram_mb():.1f} MB, time={time.time()-t0:.1f}s")

        t0 = time.time()
        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        logger.info(f"[INIT] Transformer loaded (pre-FSDP): {_param_count(self.transformer):.1f}M params, "
                     f"VRAM={_vram_mb():.1f} MB, time={time.time()-t0:.1f}s")

        shard_fn = partial(shard_model, device_id=job_config.local_rank)
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device)
        logger.info(f"[INIT] Transformer after FSDP: VRAM={_vram_mb():.1f} MB")

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            t0 = time.time()
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device=self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)
            logger.info(f"[INIT] VAE_half (wrist cameras) loaded: VRAM={_vram_mb():.1f} MB, time={time.time()-t0:.1f}s")

        # A2A / F2F modules
        self.action_encoder = None
        self.action_decoder = None
        if getattr(job_config, 'use_a2a', False):
            from modules.a2a_modules import ActionEncoder, ActionDecoder
            self.action_encoder = ActionEncoder(
                action_dim=job_config.action_dim,
            ).to(self.device, self.dtype)
            self.action_decoder = ActionDecoder(
                action_dim=job_config.action_dim,
            ).to(self.device, self.dtype)
            logger.info(f"[INIT] A2A ActionEncoder ({_param_count(self.action_encoder):.3f}M) "
                         f"+ ActionDecoder ({_param_count(self.action_decoder):.3f}M) loaded")

        logger.info(f"[INIT] All components loaded. Total VRAM={_vram_mb():.1f} MB, "
                     f"total time={time.time()-t_start_init:.1f}s")

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device),
                                          mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        t_enc_start = time.time()
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        logger.info(f"[ENCODE_OBS] num_frames={len(images)}, cam_keys={self.job_config.obs_cam_keys}")
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            raw_img = images[0][k]
            logger.info(f"[ENCODE_OBS] cam={k}: raw_shape={raw_img.shape}, target=({height_i}, {width_i})")
            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            logger.info(f"[ENCODE_OBS] cam={k}: resized tensor shape={list(history_video_k.shape)}")
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            logger.info(f"[ENCODE_OBS] robotwin_tshape: videos_high={list(videos_high.shape)}, "
                         f"videos_lr={list(videos_left_and_right.shape)}")
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(self.device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(self.device).to(self.dtype))
            logger.info(f"[ENCODE_OBS] VAE out: high={list(enc_out_high.shape)}, lr={list(enc_out_left_and_right.shape)}")
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            videos_chunk = videos.to(self.device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        logger.info(f"[ENCODE_OBS] enc_out (concat): {list(enc_out.shape)}")
        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        logger.info(f"[ENCODE_OBS] {_tensor_stats(video_latent, 'video_latent')}, "
                     f"time={time.time()-t_enc_start:.2f}s")
        return video_latent

    def _reset(self, prompt=None):
        t_reset_start = time.time()
        logger.info('=' * 60)
        logger.info('[RESET] Starting reset...')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        logger.info(f"[RESET] CFG enabled={self.use_cfg} "
                     f"(video_scale={self.job_config.guidance_scale}, action_scale={self.job_config.action_guidance_scale})")
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        # A2A / F2F history buffers
        from collections import deque
        self.state_buffer = deque(maxlen=self.job_config.frame_chunk_size)
        self.latent_buffer = deque(maxlen=self.job_config.frame_chunk_size)
        self._last_raw_actions = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        logger.info(f"[RESET] latent_size=({self.latent_height}, {self.latent_width}), "
                     f"latent_tokens/chunk={latent_token_per_chunk}, action_tokens/chunk={action_token_per_chunk}")
        logger.info(f"[RESET] attn_window={self.job_config.attn_window}, "
                     f"frame_chunk_size={self.job_config.frame_chunk_size}, "
                     f"action_per_frame={self.action_per_frame}, action_dim={self.job_config.action_dim}")
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )
        logger.info(f"[RESET] KV Cache created. VRAM={_vram_mb():.1f} MB")

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True
        logger.info(f"[RESET] action_mask: {self.action_mask.sum().item()}/{self.job_config.action_dim} channels active, "
                     f"ids={self.job_config.used_action_channel_ids}")

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            logger.info(f"[RESET] Encoding prompt: '{prompt[:80]}...'")
            t0 = time.time()
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )
            logger.info(f"[RESET] prompt_embeds={list(self.prompt_embeds.shape)}, "
                         f"neg_embeds={'None' if self.negative_prompt_embeds is None else list(self.negative_prompt_embeds.shape)}, "
                         f"time={time.time()-t0:.2f}s")

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()
        logger.info(f"[RESET] Complete. VRAM={_vram_mb():.1f} MB, time={time.time()-t_reset_start:.2f}s")
        logger.info('=' * 60)

    def _build_a2a_source(self, frame_chunk_size: int) -> torch.Tensor:
        """Build A2A flow source from state_buffer with padding (repeat earliest frame)."""
        buf = list(self.state_buffer)
        frames = []
        for item in buf:
            for f in range(item.shape[2]):
                frames.append(item[:, :, f:f + 1, :, :])
        while len(frames) < frame_chunk_size:
            frames.insert(0, frames[0].clone())
        frames = frames[-frame_chunk_size:]
        return torch.cat(frames, dim=2).to(self.device, self.dtype)

    def _build_f2f_source(self, frame_chunk_size: int) -> torch.Tensor:
        """Build F2F flow source from latent_buffer with padding (repeat earliest frame)."""
        buf = list(self.latent_buffer)
        while len(buf) < frame_chunk_size:
            buf.insert(0, buf[0].clone())
        buf = buf[-frame_chunk_size:]
        return torch.cat(buf, dim=2).to(self.device, self.dtype)

    def _infer(self, obs, frame_st_id=0):
        t_infer_start = time.time()
        frame_chunk_size = self.job_config.frame_chunk_size
        logger.info(f"[INFER] frame_st_id={frame_st_id}, frame_chunk_size={frame_chunk_size}")

        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
            # Seed F2F latent_buffer from initial observation (for i2va mode)
            if getattr(self.job_config, 'use_f2f', False) and len(self.latent_buffer) == 0:
                for f_idx in range(init_latent.shape[2]):
                    self.latent_buffer.append(init_latent[:, :, f_idx:f_idx + 1].detach().clone())

        # === F2F Video Initialization ===
        use_f2f = getattr(self.job_config, 'use_f2f', False) and len(self.latent_buffer) > 0
        if use_f2f:
            f2f_strength = self.job_config.f2f_denoising_strength
            source_latents = self._build_f2f_source(frame_chunk_size)
            if getattr(self.job_config, 'f2f_noise_std', 0) > 0:
                source_latents = source_latents + self.job_config.f2f_noise_std * torch.randn_like(source_latents)
            noise_v = torch.randn_like(source_latents)
            latents = source_latents
            logger.info(f"[INFER] F2F init from obs latent buffer (len={len(self.latent_buffer)}), "
                         f"denoising_strength={f2f_strength}")
        else:
            f2f_strength = 1.0
            noise_v = None
            latents = torch.randn(1, 48, frame_chunk_size, self.latent_height,
                                  self.latent_width, device=self.device, dtype=self.dtype)
            logger.info(f"[INFER] Standard Gaussian noise init for video latents")

        # === A2A Action Initialization ===
        use_a2a = getattr(self.job_config, 'use_a2a', False) and len(self.state_buffer) > 0
        if use_a2a:
            a2a_strength = self.job_config.a2a_denoising_strength
            source_actions = self._build_a2a_source(frame_chunk_size)
            if self.action_encoder is not None:
                source_actions = self.action_encoder(source_actions)
            if getattr(self.job_config, 'a2a_noise_std', 0) > 0:
                source_actions = source_actions + self.job_config.a2a_noise_std * torch.randn_like(source_actions)
            noise_a = torch.randn_like(source_actions)
            actions = source_actions
            logger.info(f"[INFER] A2A init from state buffer (len={len(self.state_buffer)}), "
                         f"denoising_strength={a2a_strength}")
        else:
            a2a_strength = 1.0
            noise_a = None
            actions = torch.randn(1, self.job_config.action_dim, frame_chunk_size,
                                  self.action_per_frame, 1, device=self.device, dtype=self.dtype)
            logger.info(f"[INFER] Standard Gaussian noise init for actions")

        logger.info(f"[INFER] latents={list(latents.shape)}, actions={list(actions.shape)}")

        # === Timestep setup (denoising_strength controls sigma range) ===
        video_inference_step = (
            getattr(self.job_config, 'f2f_num_inference_steps', None)
            or self.job_config.num_inference_steps
        ) if use_f2f else self.job_config.num_inference_steps
        action_inference_step = (
            getattr(self.job_config, 'a2a_num_inference_steps', None)
            or self.job_config.action_num_inference_steps
        ) if use_a2a else self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step, denoising_strength=f2f_strength)
        self.action_scheduler.set_timesteps(action_inference_step, denoising_strength=a2a_strength)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        # F2F / A2A mixing: (1-sigma_0)*source + sigma_0*noise
        if use_f2f:
            sigma_0 = self.scheduler.sigmas[0]
            latents = (1 - sigma_0) * latents + sigma_0 * noise_v
            logger.info(f"[INFER] F2F mixed with sigma_0={sigma_0:.4f}")
        if use_a2a:
            sigma_0 = self.action_scheduler.sigmas[0]
            actions = (1 - sigma_0) * actions + sigma_0 * noise_a
            logger.info(f"[INFER] A2A mixed with sigma_0={sigma_0:.4f}")

        logger.info(f"[INFER] video_steps={video_inference_step} (exec_step={video_step}), "
                     f"action_steps={action_inference_step}")
        logger.info(f"[INFER] video timesteps range: [{timesteps[0].item():.4f} -> {timesteps[-1].item():.4f}]")
        logger.info(f"[INFER] action timesteps range: [{action_timesteps[0].item():.4f} -> {action_timesteps[-1].item():.4f}]")

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.amp.autocast('cuda', dtype=self.dtype),
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            logger.info(f"[INFER] --- Video diffusion loop ({len(timesteps)} steps) ---")
            t_video_start = time.time()
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

                if i == 0 or i == len(timesteps) - 2 or last_step:
                    logger.info(f"[INFER] video step {i}/{len(timesteps)-1}: t={t.item():.4f}, "
                                 f"latent mean={latents.float().mean().item():.4f}, std={latents.float().std().item():.4f}, "
                                 f"update_cache={'YES' if last_step else 'no'}")

            t_video_end = time.time()
            logger.info(f"[INFER] Video loop done: {t_video_end - t_video_start:.2f}s, "
                         f"{_tensor_stats(latents, 'final_latents')}")

            logger.info(f"[INFER] --- Action diffusion loop ({len(action_timesteps)} steps) ---")
            t_action_start = time.time()
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

                if i == 0 or i == len(action_timesteps) - 2 or last_step:
                    logger.info(f"[INFER] action step {i}/{len(action_timesteps)-1}: t={t.item():.4f}, "
                                 f"action mean={actions.float().mean().item():.4f}, std={actions.float().std().item():.4f}, "
                                 f"update_cache={'YES' if last_step else 'no'}")

            t_action_end = time.time()
            logger.info(f"[INFER] Action loop done: {t_action_end - t_action_start:.2f}s, "
                         f"{_tensor_stats(actions, 'raw_actions')}")

        # Save pre-decode actions for i2va A2A buffer population
        self._last_raw_actions = actions.detach().clone()

        # A2A: decode flow output back to real action space
        if use_a2a and self.action_decoder is not None:
            actions = self.action_decoder(actions)
            logger.info(f"[INFER] A2A ActionDecoder applied")

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        logger.info(f"[INFER] postprocessed actions: shape={actions.shape}, "
                     f"min={actions.min():.4f}, max={actions.max():.4f}")
        torch.cuda.empty_cache()
        logger.info(f"[INFER] Chunk complete. Total time={time.time()-t_infer_start:.2f}s, VRAM={_vram_mb():.1f} MB")
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )

        # A2A / F2F: collect real observations into history buffers
        if getattr(self.job_config, 'use_a2a', False):
            self.state_buffer.append(action_model_input.detach().clone())
        if getattr(self.job_config, 'use_f2f', False) and latent_model_input is not None:
            for f_idx in range(latent_model_input.shape[2]):
                self.latent_buffer.append(
                    latent_model_input[:, :, f_idx:f_idx + 1].detach().clone()
                )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.amp.autocast('cuda', dtype=self.dtype),
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        t_gen_start = time.time()
        logger.info('#' * 60)
        logger.info(f"[GENERATE] Starting i2va generation")
        logger.info(f"[GENERATE] prompt='{self.job_config.prompt}'")
        logger.info(f"[GENERATE] num_chunks={self.job_config.num_chunks_to_infer}, "
                     f"input_img_path='{self.job_config.input_img_path}'")

        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        logger.info(f"[GENERATE] Loaded init obs: {len(init_obs['obs'])} frames, "
                     f"cam_keys={list(init_obs['obs'][0].keys())}")
        for k, v in init_obs['obs'][0].items():
            logger.info(f"[GENERATE]   {k}: shape={v.shape}, dtype={v.dtype}")

        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            logger.info(f"[GENERATE] ====== Chunk {chunk_id}/{self.job_config.num_chunks_to_infer-1} ======")
            t_chunk = time.time()
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            # i2va fallback: populate A2A/F2F buffers from generated outputs
            if getattr(self.job_config, 'use_f2f', False):
                for f_idx in range(latents.shape[2]):
                    self.latent_buffer.append(latents[:, :, f_idx:f_idx + 1].detach().clone())
            if getattr(self.job_config, 'use_a2a', False) and self._last_raw_actions is not None:
                self.state_buffer.append(self._last_raw_actions)
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
            logger.info(f"[GENERATE] Chunk {chunk_id} done in {time.time()-t_chunk:.2f}s")

        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        logger.info(f"[GENERATE] All chunks done. pred_latent={list(pred_latent.shape)}, pred_action={list(pred_action.shape)}")
        logger.info(f"[GENERATE] Total inference time={time.time()-t_gen_start:.2f}s")

        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        logger.info(f"[GENERATE] After cleanup: VRAM={_vram_mb():.1f} MB")

        logger.info(f"[GENERATE] Decoding video from latents {list(pred_latent.shape)}...")
        t_decode = time.time()
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        logger.info(f"[GENERATE] Video decoded: {len(decoded_video)} frames, "
                     f"frame_shape={decoded_video[0].shape}, time={time.time()-t_decode:.2f}s")

        out_path = os.path.join(self.save_root, "demo.mp4")
        export_to_video(decoded_video, out_path, fps=10)
        logger.info(f"[GENERATE] Video saved to {out_path}")
        logger.info(f"[GENERATE] TOTAL TIME: {time.time()-t_gen_start:.2f}s")
        logger.info('#' * 60)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
 == "__main__":
    init_logger()
    main()
