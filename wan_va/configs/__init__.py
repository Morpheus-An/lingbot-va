# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg, va_robotwin_a2a_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_libero_cfg import va_libero_cfg, va_libero_a2a_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'robotwin_a2a': va_robotwin_a2a_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_a2a': va_libero_a2a_cfg,
}