# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './visualization'

va_shared_cfg.patch_size = (1, 2, 2)

# A2A / F2F Flow Matching Settings
va_shared_cfg.use_a2a = False
va_shared_cfg.use_f2f = False
va_shared_cfg.a2a_denoising_strength = 0.5
va_shared_cfg.f2f_denoising_strength = 0.5
va_shared_cfg.a2a_noise_std = 0.1
va_shared_cfg.f2f_noise_std = 0.0
va_shared_cfg.a2a_num_inference_steps = None
va_shared_cfg.f2f_num_inference_steps = None