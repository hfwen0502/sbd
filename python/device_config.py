"""
Device configuration helper for SBD Python bindings.

This module provides utilities to easily switch between CPU and GPU execution
without changing user code.
"""

import subprocess


class DeviceConfig:
    """
    Helper class to configure CPU vs GPU execution for SBD calculations.
    
    Usage:
        # Auto-detect (uses GPU if available)
        config = DeviceConfig.auto()
        
        # Force CPU
        config = DeviceConfig.cpu()
        
        # Force GPU with specific settings
        config = DeviceConfig.gpu(max_memory_gb=16)
        
        # Apply to SBD configuration
        sbd_config = sbd.TPB_SBD()
        config.apply(sbd_config)
    """
    
    def __init__(self, device: str = 'cpu',
                 use_precalculated_dets: bool = True,
                 max_memory_gb: int = -1,
                 use_gpu: bool | None = None):
        """
        Initialize device configuration.

        Args:
            device: Backend device key — 'cpu', 'gpu' (NVHPC Thrust),
                'gpu-nvidia-omp' (clang OpenMP-offload, NVIDIA), or any
                alias known to ``sbd._device_aliases``. Default 'cpu'.
            use_precalculated_dets: Use precalculated determinants (GPU only)
            max_memory_gb: Maximum GPU memory in GB (-1 = auto)
            use_gpu: Deprecated boolean. If supplied without ``device``,
                ``True`` maps to 'gpu' (Thrust) and ``False`` to 'cpu'
                for backward compatibility with pre-OMP-offload code.
        """
        if use_gpu is not None and device == 'cpu':
            # Legacy boolean path
            device = 'gpu' if use_gpu else 'cpu'
        self.device = device
        self.use_gpu = device != 'cpu'
        self.use_precalculated_dets = use_precalculated_dets
        self.max_memory_gb = max_memory_gb
    
    @classmethod
    def auto(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """
        Auto-detect GPU availability and use it if available.
        
        Args:
            max_memory_gb: Maximum GPU memory in GB (-1 = auto)
            
        Returns:
            DeviceConfig configured for GPU if available, CPU otherwise
        """
        # Check if CUDA or HIP is available
        has_cuda = cls._check_cuda()
        has_hip = cls._check_hip()
        
        use_gpu = has_cuda or has_hip

        if use_gpu:
            print(f"GPU detected ({'CUDA' if has_cuda else 'HIP'}), using GPU acceleration")
        else:
            print("No GPU detected, using CPU")

        device = 'gpu' if use_gpu else 'cpu'
        return cls(device=device, max_memory_gb=max_memory_gb)

    @classmethod
    def cpu(cls) -> 'DeviceConfig':
        """Force CPU execution."""
        return cls(device='cpu')

    @classmethod
    def gpu(cls, use_precalculated_dets: bool = True,
            max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force NVHPC Thrust GPU execution.

        Requires SBD compiled with THRUST (the ``_core_gpu`` extension,
        i.e. ``SBD_BUILD_BACKEND=gpu`` or ``=both``).
        """
        return cls(device='gpu',
                   use_precalculated_dets=use_precalculated_dets,
                   max_memory_gb=max_memory_gb)

    @classmethod
    def gpu_omp(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force OpenMP-offload GPU execution (NVIDIA).

        Requires SBD compiled with the OMP-offload backend (the
        ``_core_gpu_omp_nvidia`` extension, i.e.
        ``SBD_BUILD_BACKEND=gpu_omp_nvidia``). Alias for
        :meth:`gpu_nvidia_omp`.
        """
        return cls.gpu_nvidia_omp(max_memory_gb=max_memory_gb)

    @classmethod
    def gpu_nvidia_omp(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force OpenMP-offload GPU execution on NVIDIA hardware.

        See :meth:`gpu_omp` for details. Use this name when you want to
        be explicit (e.g. when AMD OMP-offload is also configured).
        """
        return cls(device='gpu-nvidia-omp', max_memory_gb=max_memory_gb)
    
    # Cached detection results (None = not yet checked)
    _cuda_cache: bool | None = None
    _hip_cache: bool | None = None

    @classmethod
    def _check_cuda(cls) -> bool:
        """Check if CUDA is available (cached)."""
        if cls._cuda_cache is not None:
            return cls._cuda_cache
        try:
            result = subprocess.run(
                ['nvidia-smi'], capture_output=True, timeout=2
            )
            cls._cuda_cache = result.returncode == 0
        except Exception:
            cls._cuda_cache = False
        return cls._cuda_cache

    @classmethod
    def _check_hip(cls) -> bool:
        """Check if HIP/ROCm is available (cached)."""
        if cls._hip_cache is not None:
            return cls._hip_cache
        try:
            result = subprocess.run(
                ['rocm-smi'], capture_output=True, timeout=2
            )
            cls._hip_cache = result.returncode == 0
        except Exception:
            cls._hip_cache = False
        return cls._hip_cache
    
    def apply(self, sbd_config) -> None:
        """
        Apply device configuration to an SBD TPB_SBD configuration object.
        
        Args:
            sbd_config: sbd.TPB_SBD configuration object
        """
        # GPU-specific parameters are only available if compiled with THRUST
        if self.use_gpu:
            try:
                sbd_config.use_precalculated_dets = self.use_precalculated_dets
                sbd_config.max_memory_gb_for_determinants = self.max_memory_gb
            except AttributeError:
                print("WARNING: GPU parameters not available. "
                      "SBD may not be compiled with THRUST support.")
                print("Falling back to CPU execution.")
    
    def __repr__(self) -> str:
        if self.device == 'cpu':
            return "DeviceConfig(CPU)"
        return (f"DeviceConfig(device={self.device!r}, "
                f"precalc_dets={self.use_precalculated_dets}, "
                f"max_mem={self.max_memory_gb}GB)")


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': DeviceConfig._check_cuda(),
        'hip_available': DeviceConfig._check_hip(),
        'gpu_available': False,
        'gpu_type': None,
        'gpu_count': 0
    }
    
    if info['cuda_available']:
        info['gpu_available'] = True
        info['gpu_type'] = 'CUDA'
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                info['gpu_count'] = len([l for l in result.stdout.split('\n') if l.strip()])
        except Exception:
            pass

    elif info['hip_available']:
        info['gpu_available'] = True
        info['gpu_type'] = 'HIP/ROCm'
        try:
            result = subprocess.run(
                ['rocm-smi', '--showid'],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                info['gpu_count'] = len([l for l in result.stdout.split('\n')
                                        if 'GPU' in l])
        except Exception:
            pass
    
    return info


def print_device_info():
    """Print information about available compute devices."""
    info = get_device_info()
    
    print("="*60)
    print("SBD Device Information")
    print("="*60)
    
    if info['gpu_available']:
        print(f"✓ GPU Available: {info['gpu_type']}")
        if info['gpu_count'] > 0:
            print(f"  GPU Count: {info['gpu_count']}")
    else:
        print("✗ No GPU detected")
    
    print(f"✓ CPU Available: Always")
    print("="*60)
