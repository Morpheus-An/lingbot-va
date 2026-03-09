# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from copy import deepcopy

from easydict import EasyDict

from .shared_config import va_shared_cfg

# ---------------------------------------------------------------------------
# LIBERO base config (single-arm Franka, 2 cameras)
# ---------------------------------------------------------------------------
va_libero_cfg = EasyDict(__name__='Config: VA libero')
va_libero_cfg.update(va_shared_cfg)
va_libero_cfg.infer_mode = 'server'

va_libero_cfg.wan22_pretrained_model_name_or_path = "~/checkpoints/lingbot-va-base"

va_libero_cfg.attn_window = 30
va_libero_cfg.frame_chunk_size = 4
va_libero_cfg.env_type = 'none'

va_libero_cfg.height = 256
va_libero_cfg.width = 256
va_libero_cfg.action_dim = 30
va_libero_cfg.action_per_frame = 20
va_libero_cfg.obs_cam_keys = [
    'agentview_rgb',
    'eye_in_hand_rgb',
]
va_libero_cfg.guidance_scale = 5
va_libero_cfg.action_guidance_scale = 1

va_libero_cfg.num_inference_steps = 5
va_libero_cfg.video_exec_step = -1
va_libero_cfg.action_num_inference_steps = 10

va_libero_cfg.snr_shift = 5.0
va_libero_cfg.action_snr_shift = 1.0

# LIBERO action mapping: 7D -> 30D universal format
# [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
#  -> channels 0-5 (EEF 6D) + channel 14 (gripper)
va_libero_cfg.used_action_channel_ids = list(range(0, 6)) + [14]
inverse_used_action_channel_ids = [
    len(va_libero_cfg.used_action_channel_ids)
] * va_libero_cfg.action_dim
for i, j in enumerate(va_libero_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_libero_cfg.action_norm_method = 'quantiles'
# Placeholder — will be computed from LIBERO data via normalize.py
va_libero_cfg.norm_stat = {
    "q01": [0.0] * 7 + [0.0] * 23,
    "q99": [1.0] * 7 + [0.0] * 21 + [1.0, 1.0],
}

# ---- Training-specific params ----
va_libero_cfg.lr = 1e-5
va_libero_cfg.weight_decay = 0.01
va_libero_cfg.train_steps = 4000
va_libero_cfg.warmup_steps = 200
va_libero_cfg.grad_clip_norm = 2.0
va_libero_cfg.cfg_dropout_rate = 0.1
va_libero_cfg.save_every = 500
va_libero_cfg.log_every = 50
va_libero_cfg.sequence_length = 100000
va_libero_cfg.lambda_ae = 1.0
va_libero_cfg.lambda_ic = 1.0
va_libero_cfg.lambda_decode = 0.5

# LIBERO state mapping (for A2A source): 7 joint + 1 gripper = 8D
# Map to the same 30D format using the first 7 channels + channel 14
va_libero_cfg.state_dim = 8
va_libero_cfg.used_state_channel_ids = list(range(0, 7)) + [14]

# ---------------------------------------------------------------------------
# LIBERO A2A / F2F variant
# ---------------------------------------------------------------------------
va_libero_a2a_cfg = deepcopy(va_libero_cfg)
va_libero_a2a_cfg.__name__ = 'Config: VA libero A2A'
va_libero_a2a_cfg.use_a2a = True
va_libero_a2a_cfg.use_f2f = True
va_libero_a2a_cfg.a2a_denoising_strength = 0.5
va_libero_a2a_cfg.f2f_denoising_strength = 0.5
va_libero_a2a_cfg.a2a_noise_std = 0.1
va_libero_a2a_cfg.f2f_noise_std = 0.0
va_libero_a2a_cfg.a2a_num_inference_steps = 5
va_libero_a2a_cfg.f2f_num_inference_steps = 3
