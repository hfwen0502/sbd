from setuptools import setup, Extension
import sys
import os
import subprocess
import pybind11


class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()


def get_mpi4py_include():
    try:
        import mpi4py
        return mpi4py.get_include()
    except (ImportError, AttributeError):
        import site
        for site_dir in site.getsitepackages():
            mpi4py_inc = os.path.join(site_dir, 'mpi4py', 'include')
            if os.path.exists(mpi4py_inc):
                return mpi4py_inc
        return None


def get_mpi_config():
    mpi_home = os.environ.get('MPI_HOME', None)
    if mpi_home:
        mpi_include = os.path.join(mpi_home, 'include')
        mpi_lib = os.path.join(mpi_home, 'lib')
        print(f"Using MPI from MPI_HOME: {mpi_home}")
        return [mpi_include], [mpi_lib], ['mpi']
    try:
        compile_flags = subprocess.check_output(['mpicc', '--showme:compile'],
                                                universal_newlines=True).strip().split()
        link_flags = subprocess.check_output(['mpicc', '--showme:link'],
                                             universal_newlines=True).strip().split()
        include_dirs = [flag[2:] for flag in compile_flags if flag.startswith('-I')]
        library_dirs = [flag[2:] for flag in link_flags if flag.startswith('-L')]
        libraries = [flag[2:] for flag in link_flags if flag.startswith('-l')]
        print("Using MPI detected from mpicc")
        return include_dirs, library_dirs, libraries
    except Exception as e:
        print(f"Error: Could not detect MPI. Please set MPI_HOME environment variable.")
        print(f"Error details: {e}")
        sys.exit(1)


def find_nvidia_hpc_sdk():
    nvhpc_home = os.environ.get('NVHPC_HOME', None)
    if nvhpc_home:
        nvcxx_path = os.path.join(nvhpc_home, 'bin', 'nvc++')
        if os.path.exists(nvcxx_path):
            print(f"Found NVIDIA HPC SDK at: {nvhpc_home}")
            nvhpc_bin = os.path.join(nvhpc_home, 'bin')
            current_path = os.environ.get('PATH', '')
            if nvhpc_bin not in current_path:
                os.environ['PATH'] = f"{nvhpc_bin}:{current_path}"
            return nvcxx_path, True
        else:
            print(f"Warning: NVHPC_HOME set to {nvhpc_home} but nvc++ not found")
    import shutil
    nvcxx_path = shutil.which('nvc++')
    if nvcxx_path:
        print(f"Found nvc++ in PATH: {nvcxx_path}")
        return nvcxx_path, True
    return None, False


def _llvm_host_triple_subdir(llvm_home):
    """Pick the host-runtime subdir under LLVM_HOME/lib/.

    LLVM with -DLLVM_RUNTIME_TARGETS=default installs the host runtime
    libs (libomp, libomptarget) into lib/<triple>/ where <triple>
    matches the host arch — x86_64-unknown-linux-gnu on Intel/AMD x86,
    aarch64-unknown-linux-gnu on ARM (Grace, Neoverse, etc.).

    We prefer the platform-derived guess but fall back to whichever
    triple subdir actually contains libomptarget.so so non-standard
    installs still work.
    """
    import glob, platform
    arch_to_triple = {
        'x86_64':  'x86_64-unknown-linux-gnu',
        'aarch64': 'aarch64-unknown-linux-gnu',
    }
    primary = arch_to_triple.get(platform.machine())
    candidates = [primary] if primary else []
    # Fallback: scan lib/ for any *-unknown-linux-gnu subdir with omptarget
    for d in glob.glob(os.path.join(llvm_home, 'lib', '*-unknown-linux-gnu')):
        name = os.path.basename(d)
        if name not in candidates:
            candidates.append(name)
    for triple in candidates:
        path = os.path.join(llvm_home, 'lib', triple)
        if glob.glob(os.path.join(path, 'libomptarget.so*')):
            return path
    # Last resort — return primary path even if libomptarget isn't there;
    # the caller logs a clear "missing libomptarget" warning in that case.
    return os.path.join(llvm_home, 'lib', primary or 'x86_64-unknown-linux-gnu')


def find_llvm_offload():
    """Return (clang++_path, has_llvm, host_lib_dir) for an LLVM-with-offload install.

    Looks at LLVM_HOME and verifies:
      1. bin/clang++ exists,
      2. lib/<host-triple>/libomptarget.so* (host runtime dispatcher) is
         present (host triple auto-detected — x86_64 or aarch64),
      3. lib/nvptx64-nvidia-cuda/libomptarget-nvptx.bc (device-side
         bitcode for NVIDIA) is present.

    All three indicate a working OMP-offload install. Stock distro clang
    (RHEL clang-19, etc.) usually has #1 and #2 but not #3 — without the
    device bitcode omp_get_num_devices() returns 0 and offload silently
    falls back to host. See .github/SETUP_LLVM_OFFLOAD.txt for how to
    build LLVM trunk with the Offload runtime enabled.

    Note: in older LLVM (<=18 or so) the CUDA RTL was a separate
    libomptarget.rtl.cuda*.so file. In LLVM trunk this has been merged
    into libomptarget.so itself, so we don't search for the RTL file.
    """
    llvm_home = os.environ.get('LLVM_HOME', None)
    if not llvm_home:
        return None, False, None
    clangxx = os.path.join(llvm_home, 'bin', 'clang++')
    if not os.path.exists(clangxx):
        print(f"Warning: LLVM_HOME={llvm_home} but {clangxx} not found")
        return None, False, None
    import glob
    host_lib = _llvm_host_triple_subdir(llvm_home)
    if not glob.glob(os.path.join(host_lib, 'libomptarget.so*')):
        print(f"Warning: LLVM_HOME={llvm_home} has clang++ but no "
              f"libomptarget.so* in {host_lib}; build LLVM with "
              f"LLVM_RUNTIME_TARGETS including 'default' (see "
              f".github/SETUP_LLVM_OFFLOAD.txt). Skipping OMP-offload backend.")
        return None, False, None
    nvptx_bc = os.path.join(llvm_home, 'lib', 'nvptx64-nvidia-cuda',
                            'libomptarget-nvptx.bc')
    if not os.path.exists(nvptx_bc):
        print(f"Warning: LLVM_HOME={llvm_home} is missing the NVIDIA "
              f"device-side runtime ({nvptx_bc}); GPU offload would "
              f"compile but produce no device kernels. Build LLVM with "
              f"LLVM_RUNTIME_TARGETS including 'nvptx64-nvidia-cuda'. "
              f"Skipping OMP-offload backend.")
        return None, False, None
    print(f"Found LLVM-with-offload at: {llvm_home} (host triple: {os.path.basename(host_lib)})")
    return clangxx, True, host_lib


# Get MPI configuration
mpi_includes, mpi_lib_dirs, mpi_libs = get_mpi_config()

# Get mpi4py include path
mpi4py_inc = get_mpi4py_include()
if not mpi4py_inc:
    print("Warning: Could not find mpi4py include path")

# Build include/library directories.
# SBD's C++ headers come from the vendored upstream submodule.
# After cloning the parent repo, run:  git submodule update --init --recursive
SBD_UPSTREAM_INCLUDE = os.path.join('vendor', 'sbd-upstream', 'include')
if not os.path.isdir(SBD_UPSTREAM_INCLUDE):
    print(f"Error: {SBD_UPSTREAM_INCLUDE} not found.")
    print("Run: git submodule update --init --recursive")
    sys.exit(1)
include_dirs = [get_pybind_include(), SBD_UPSTREAM_INCLUDE] + mpi_includes
if mpi4py_inc:
    include_dirs.append(mpi4py_inc)

library_dirs = mpi_lib_dirs.copy()

blas_lib_path = os.environ.get('BLAS_LIB_PATH', None)
if blas_lib_path:
    library_dirs.append(blas_lib_path)
    print(f"Using BLAS from: {blas_lib_path}")
else:
    print("Warning: BLAS_LIB_PATH not set. Assuming BLAS is in system path.")

blas_libs = os.environ.get('BLAS_LIBS', 'openblas').split(',')
print(f"Using BLAS libraries: {blas_libs}")

libraries = mpi_libs + blas_libs

# RPATH so libraries are found at runtime without LD_LIBRARY_PATH
extra_link_args = ['-fopenmp']
for lib_dir in library_dirs:
    extra_link_args.append(f'-Wl,--rpath,{lib_dir}')
print(f"RPATH will be set to: {library_dirs}")

# Detect GPU compilers
gpu_compiler, has_nvhpc = find_nvidia_hpc_sdk()
omp_clang, has_llvm, llvm_host_lib = find_llvm_offload()

# Determine which backends to build
build_backend = os.environ.get('SBD_BUILD_BACKEND', 'auto').lower()

# Defaults (each branch overrides the ones it sets to True)
build_cpu = False
build_gpu = False
build_gpu_omp_nvidia = False

if build_backend == 'auto':
    build_cpu = True
    build_gpu = has_nvhpc
    if build_gpu:
        print("\nAuto-detected nvc++ - will build both CPU and GPU backends")
    else:
        print("\nnvc++ not found - will build CPU backend only")
    if has_llvm:
        print("Note: LLVM_HOME with offload runtime detected. To build the "
              "OMP-offload backend run a separate pip install with "
              "SBD_BUILD_BACKEND=gpu_omp_nvidia (must be built alone, "
              "see setup.py header for why).")
elif build_backend == 'cpu':
    build_cpu = True
    print("\nBuilding CPU backend only (SBD_BUILD_BACKEND=cpu)")
elif build_backend == 'gpu':
    build_gpu = True
    print("\nBuilding GPU backend only (SBD_BUILD_BACKEND=gpu)")
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
elif build_backend == 'both':
    build_cpu = True
    build_gpu = True
    print("\nBuilding both CPU and GPU backends (SBD_BUILD_BACKEND=both)")
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
elif build_backend == 'gpu_omp_nvidia':
    # Stand-alone build: this mode only emits _core_gpu_omp_nvidia.so.
    # CPU and Thrust GPU backends use different compilers (gcc / nvc++),
    # so trying to build them together with clang++ in the same setup()
    # call breaks distutils' single-CC/CXX assumption. Run separate
    # pip installs for those if you also want them.
    build_gpu_omp_nvidia = True
    print("\nBuilding GPU OMP-offload backend only "
          "(SBD_BUILD_BACKEND=gpu_omp_nvidia)")
    if not has_llvm:
        print("Error: gpu_omp_nvidia requires LLVM_HOME set to an LLVM "
              "trunk install with the offload runtime built.")
        print("See .github/SETUP_LLVM_OFFLOAD.txt for the build recipe.")
        sys.exit(1)
else:
    print(f"Error: Invalid SBD_BUILD_BACKEND='{build_backend}'")
    print("Valid values: auto, cpu, gpu, both, gpu_omp_nvidia")
    sys.exit(1)

ext_modules = []

if build_cpu:
    print("\nConfiguring CPU backend (_core_cpu)")
    import platform
    if platform.system() == 'Darwin':
        omp_inc = '/opt/homebrew/opt/libomp/include'
        omp_lib = '/opt/homebrew/opt/libomp/lib'
        openblas_lib = '/opt/homebrew/opt/openblas/lib'
        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-std=c++17', '-Xpreprocessor', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu', f'-I{omp_inc}',
        ]
        cpu_link_args = [f'-L{omp_lib}', f'-L{openblas_lib}', '-lomp']
        cpu_inc = include_dirs + [omp_inc]
        cpu_lib_dirs = library_dirs + [omp_lib, openblas_lib]
        cpu_libs = libraries + ['omp']
    else:
        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-DOMPI_SKIP_MPICXX',
            '-std=c++17', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu',
        ]
        cpu_link_args = extra_link_args
        cpu_inc = include_dirs
        cpu_lib_dirs = library_dirs
        cpu_libs = libraries

    cpu_ext = Extension(
        'sbd._core_cpu',
        ['python/bindings.cpp'],
        include_dirs=cpu_inc,
        libraries=cpu_libs,
        library_dirs=cpu_lib_dirs,
        language='c++',
        extra_compile_args=cpu_compile_args,
        extra_link_args=cpu_link_args,
    )
    ext_modules.append(cpu_ext)

if build_gpu:
    print("\nConfiguring GPU backend (_core_gpu)")
    if not gpu_compiler:
        print("Error: GPU backend requested but nvc++ not found")
        sys.exit(1)
    print(f"Using compiler: {gpu_compiler}")

    gpu_ext = Extension(
        'sbd._core_gpu',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=[
            '-DSBD_THRUST',
            '-DSBD_TRADMODE',
            '-mp',
            '-cuda',
            '-fast',
            '-Minfo=accel',
            '--diag_suppress=declared_but_not_referenced,set_but_not_used',
            '-fmax-errors=0',
            '-fPIC',
            '-gpu=sm_90',
            '-DSBD_MODULE_NAME=_core_gpu',
        ],
        extra_link_args=extra_link_args + ['-mp', '-cuda', '-cudalib'],
    )
    ext_modules.append(gpu_ext)


if build_gpu_omp_nvidia:
    print("\nConfiguring GPU OMP-offload backend (_core_gpu_omp_nvidia)")
    print(f"Using compiler: {omp_clang}")

    # Drive the build through clang++. Distutils picks compiler from
    # CC / CXX / LDSHARED; only one compiler choice per setup() call,
    # which is why this mode must be invoked alone.
    os.environ['CC']       = omp_clang
    os.environ['CXX']      = omp_clang
    os.environ['LDSHARED'] = f'{omp_clang} -shared'

    # Clear sysconfig-inherited flags. RHEL CPython ships with
    # -fcf-protection={branch,return} which are x86-only and clang
    # rejects when compiling the nvptx offload pass. We pass our own
    # full flag list via extra_compile_args below.
    os.environ['CFLAGS']   = ''
    os.environ['CXXFLAGS'] = ''
    # Same for distutils' default optimization flag set
    os.environ['CPPFLAGS'] = ''

    offload_arch = os.environ.get('SBD_OFFLOAD_ARCH_NVIDIA', 'sm_90')
    # Host-arch-aware: x86_64 -> x86_64-unknown-linux-gnu/, aarch64 -> aarch64-...
    triple_lib   = llvm_host_lib

    gpu_omp_nvidia_ext = Extension(
        'sbd._core_gpu_omp_nvidia',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        # Add the LLVM runtime triple subdir to library search; rpath
        # below ensures the resulting .so finds libomptarget at import.
        library_dirs=library_dirs + [triple_lib],
        language='c++',
        extra_compile_args=[
            '-O3', '-std=c++17', '-fPIC',
            '-fopenmp',
            '-fopenmp-targets=nvptx64-nvidia-cuda',
            f'--offload-arch={offload_arch}',
            '-fopenmp-offload-mandatory',
            '-foffload-lto',
            '-DSBD_TRADMODE',
            '-DUSE_GPU',
            '-DUSE_OMP_OFFLOAD',
            '-DOMPI_SKIP_MPICXX',
            '-DSBD_MODULE_NAME=_core_gpu_omp_nvidia',
        ],
        extra_link_args=extra_link_args + [
            '-fopenmp',
            f'--offload-arch={offload_arch}',
            '-foffload-lto',
            f'-Wl,-rpath,{triple_lib}',
        ],
    )
    ext_modules.append(gpu_omp_nvidia_ext)


setup(
    name='sbd',
    version='1.3.0',
    author='Tomonori Shirakawa',
    author_email='',
    description='Python bindings for Selected Basis Diagonalization (SBD) library',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/hfwen0502/sbd',
    packages=['sbd'],
    package_dir={'sbd': 'python'},
    ext_modules=ext_modules,
    install_requires=[
        'pybind11>=2.6.0',
        'mpi4py>=3.0.0',
        'numpy>=1.19.0',
    ],
    python_requires='>=3.10',
    zip_safe=False,
)

print("\nSetup complete!")
if build_cpu:
    print("  - CPU backend:                    sbd._core_cpu")
if build_gpu:
    print("  - GPU backend (NVHPC Thrust):     sbd._core_gpu")
if build_gpu_omp_nvidia:
    print("  - GPU backend (OMP-offload NV):   sbd._core_gpu_omp_nvidia")
print()
