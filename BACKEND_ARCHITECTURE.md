# Backend Architecture Design: PyTorch-Style Factory Pattern

## Overview

This document proposes a PyTorch-inspired backend architecture for SBD's Python bindings. The key insight from PyTorch is:

**Everything is resolved in the C++ layer through polymorphism, factory registry, compile-time feature flags, and dynamic linking.**

## PyTorch's Actual Architecture

### Key Principles from PyTorch

1. **C++ Polymorphism**: Abstract base class with virtual methods
2. **Factory Registry**: Map of backend names to factory functions
3. **Compile-Time Feature Flags**: `#ifdef` to conditionally compile backends
4. **Dynamic Linking**: Link to actual libraries (MPI, NCCL, etc.) at build time
5. **Python Passes String**: Python just passes backend name, C++ does everything
6. **No Runtime Patching**: No monkey patching, no separate Python modules
7. **Single C++ Module**: Everything in one compiled extension

### PyTorch Example

```cpp
// PyTorch's approach
class ProcessGroup {  // Abstract base
public:
    virtual void allreduce(...) = 0;
};

class ProcessGroupNCCL : public ProcessGroup { ... };  // GPU
class ProcessGroupMPI  : public ProcessGroup { ... };  // CPU
class ProcessGroupGloo : public ProcessGroup { ... };  // CPU fallback

// Factory
std::shared_ptr<ProcessGroup> createProcessGroup(std::string backend) {
    if (backend == "nccl") return std::make_shared<ProcessGroupNCCL>();
    if (backend == "mpi")  return std::make_shared<ProcessGroupMPI>();
    // ...
}
```

Python just calls:
```python
torch.distributed.init_process_group(backend="nccl")  # String passed to C++
```

## SBD Backend Architecture

### 1. Abstract Backend Interface

```cpp
// File: sbd/tpb/backend_interface.h

#ifndef SBD_TPB_BACKEND_INTERFACE_H
#define SBD_TPB_BACKEND_INTERFACE_H

#include <memory>
#include <string>
#include <vector>
#include <mpi.h>
#include "sbd/sbd.h"

namespace sbd {
namespace tpb {

/**
 * Abstract interface for TPB diagonalization backends
 * Inspired by PyTorch's ProcessGroup interface
 */
class DiagBackend {
public:
    virtual ~DiagBackend() = default;
    
    /**
     * Main diagonalization function - pure virtual
     */
    virtual void diag(
        MPI_Comm comm,
        const SBD& sbd_data,
        const FCIDump& fcidump,
        const std::vector<std::vector<size_t>>& adet,
        const std::vector<std::vector<size_t>>& bdet,
        const std::string& loadname,
        const std::string& savename,
        double& energy,
        std::vector<double>& density,
        std::vector<std::vector<size_t>>& co_adet,
        std::vector<std::vector<size_t>>& co_bdet,
        std::vector<std::vector<double>>& one_p_rdm,
        std::vector<std::vector<double>>& two_p_rdm
    ) = 0;
    
    /**
     * Backend metadata
     */
    virtual std::string name() const = 0;
    virtual std::string device_type() const = 0;  // "cpu", "cuda", "hip"
    virtual bool is_available() const = 0;
};

} // namespace tpb
} // namespace sbd

#endif // SBD_TPB_BACKEND_INTERFACE_H
```

### 2. CPU Backend Implementation

```cpp
// File: sbd/tpb/backend_cpu.h

#ifndef SBD_TPB_BACKEND_CPU_H
#define SBD_TPB_BACKEND_CPU_H

#include "sbd/tpb/backend_interface.h"

namespace sbd {
namespace tpb {

/**
 * CPU backend - always available
 * Uses existing CPU implementation
 */
class DiagBackendCPU : public DiagBackend {
public:
    void diag(
        MPI_Comm comm,
        const SBD& sbd_data,
        const FCIDump& fcidump,
        const std::vector<std::vector<size_t>>& adet,
        const std::vector<std::vector<size_t>>& bdet,
        const std::string& loadname,
        const std::string& savename,
        double& energy,
        std::vector<double>& density,
        std::vector<std::vector<size_t>>& co_adet,
        std::vector<std::vector<size_t>>& co_bdet,
        std::vector<std::vector<double>>& one_p_rdm,
        std::vector<std::vector<double>>& two_p_rdm
    ) override;
    
    std::string name() const override { return "cpu"; }
    std::string device_type() const override { return "cpu"; }
    bool is_available() const override { return true; }
};

} // namespace tpb
} // namespace sbd

#endif // SBD_TPB_BACKEND_CPU_H
```

```cpp
// File: sbd/tpb/backend_cpu.cpp

#include "sbd/tpb/backend_cpu.h"
#include "sbd/tpb/diag_impl.h"  // Existing implementation

namespace sbd {
namespace tpb {

void DiagBackendCPU::diag(
    MPI_Comm comm,
    const SBD& sbd_data,
    const FCIDump& fcidump,
    const std::vector<std::vector<size_t>>& adet,
    const std::vector<std::vector<size_t>>& bdet,
    const std::string& loadname,
    const std::string& savename,
    double& energy,
    std::vector<double>& density,
    std::vector<std::vector<size_t>>& co_adet,
    std::vector<std::vector<size_t>>& co_bdet,
    std::vector<std::vector<double>>& one_p_rdm,
    std::vector<std::vector<double>>& two_p_rdm
) {
    // Call existing CPU implementation
    // This is the current sbd::tpb::diag() code without GPU parts
    diag_cpu_impl(comm, sbd_data, fcidump, adet, bdet,
                  loadname, savename, energy, density,
                  co_adet, co_bdet, one_p_rdm, two_p_rdm);
}

} // namespace tpb
} // namespace sbd
```

### 3. CUDA Backend Implementation (Compile-Time Conditional)

```cpp
// File: sbd/tpb/backend_cuda.h

#ifndef SBD_TPB_BACKEND_CUDA_H
#define SBD_TPB_BACKEND_CUDA_H

#include "sbd/tpb/backend_interface.h"

// Only compile if CUDA support is enabled
#ifdef SBD_THRUST
#ifdef __CUDACC__

#include <cuda_runtime.h>

namespace sbd {
namespace tpb {

/**
 * CUDA backend - available only if compiled with CUDA support
 * Uses THRUST for GPU acceleration
 */
class DiagBackendCUDA : public DiagBackend {
private:
    int device_id_;
    
public:
    explicit DiagBackendCUDA(int device_id = -1);
    
    void diag(
        MPI_Comm comm,
        const SBD& sbd_data,
        const FCIDump& fcidump,
        const std::vector<std::vector<size_t>>& adet,
        const std::vector<std::vector<size_t>>& bdet,
        const std::string& loadname,
        const std::string& savename,
        double& energy,
        std::vector<double>& density,
        std::vector<std::vector<size_t>>& co_adet,
        std::vector<std::vector<size_t>>& co_bdet,
        std::vector<std::vector<double>>& one_p_rdm,
        std::vector<std::vector<double>>& two_p_rdm
    ) override;
    
    std::string name() const override { return "cuda"; }
    std::string device_type() const override { return "cuda"; }
    bool is_available() const override;
};

} // namespace tpb
} // namespace sbd

#endif // __CUDACC__
#endif // SBD_THRUST

#endif // SBD_TPB_BACKEND_CUDA_H
```

```cpp
// File: sbd/tpb/backend_cuda.cpp

#include "sbd/tpb/backend_cuda.h"

#ifdef SBD_THRUST
#ifdef __CUDACC__

#include "sbd/tpb/diag_impl.h"

namespace sbd {
namespace tpb {

DiagBackendCUDA::DiagBackendCUDA(int device_id) : device_id_(device_id) {
    if (device_id_ < 0) {
        // Auto-select device based on MPI rank
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        device_id_ = mpi_rank % num_devices;
    }
    cudaSetDevice(device_id_);
}

void DiagBackendCUDA::diag(
    MPI_Comm comm,
    const SBD& sbd_data,
    const FCIDump& fcidump,
    const std::vector<std::vector<size_t>>& adet,
    const std::vector<std::vector<size_t>>& bdet,
    const std::string& loadname,
    const std::string& savename,
    double& energy,
    std::vector<double>& density,
    std::vector<std::vector<size_t>>& co_adet,
    std::vector<std::vector<size_t>>& co_bdet,
    std::vector<std::vector<double>>& one_p_rdm,
    std::vector<std::vector<double>>& two_p_rdm
) {
    // Call GPU implementation with THRUST
    diag_cuda_impl(comm, sbd_data, fcidump, adet, bdet,
                   loadname, savename, energy, density,
                   co_adet, co_bdet, one_p_rdm, two_p_rdm);
}

bool DiagBackendCUDA::is_available() const {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

} // namespace tpb
} // namespace sbd

#endif // __CUDACC__
#endif // SBD_THRUST
```

### 4. HIP Backend Implementation (Compile-Time Conditional)

```cpp
// File: sbd/tpb/backend_hip.h

#ifndef SBD_TPB_BACKEND_HIP_H
#define SBD_TPB_BACKEND_HIP_H

#include "sbd/tpb/backend_interface.h"

// Only compile if HIP support is enabled
#ifdef SBD_THRUST
#ifndef __CUDACC__  // HIP, not CUDA

#include <hip/hip_runtime.h>

namespace sbd {
namespace tpb {

class DiagBackendHIP : public DiagBackend {
private:
    int device_id_;
    
public:
    explicit DiagBackendHIP(int device_id = -1);
    
    void diag(...) override;
    
    std::string name() const override { return "hip"; }
    std::string device_type() const override { return "hip"; }
    bool is_available() const override;
};

} // namespace tpb
} // namespace sbd

#endif // !__CUDACC__
#endif // SBD_THRUST

#endif // SBD_TPB_BACKEND_HIP_H
```

### 5. Backend Factory with Registry

```cpp
// File: sbd/tpb/backend_factory.h

#ifndef SBD_TPB_BACKEND_FACTORY_H
#define SBD_TPB_BACKEND_FACTORY_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

#include "sbd/tpb/backend_interface.h"
#include "sbd/tpb/backend_cpu.h"

// Conditionally include GPU backends
#ifdef SBD_THRUST
#ifdef __CUDACC__
#include "sbd/tpb/backend_cuda.h"
#else
#include "sbd/tpb/backend_hip.h"
#endif
#endif

namespace sbd {
namespace tpb {

/**
 * Factory for creating backend instances
 * Similar to PyTorch's backend creation mechanism
 * 
 * Key features:
 * - Singleton pattern
 * - Registry of backend creators
 * - Compile-time conditional registration
 * - Runtime availability checking
 */
class BackendFactory {
public:
    using BackendCreator = std::function<std::shared_ptr<DiagBackend>()>;
    
private:
    std::map<std::string, BackendCreator> registry_;
    
    // Private constructor for singleton
    BackendFactory() {
        // Always register CPU backend
        register_backend("cpu", []() {
            return std::make_shared<DiagBackendCPU>();
        });
        
        // Conditionally register GPU backends at compile time
#ifdef SBD_THRUST
#ifdef __CUDACC__
        // CUDA backend
        register_backend("cuda", []() {
            return std::make_shared<DiagBackendCUDA>();
        });
#else
        // HIP backend
        register_backend("hip", []() {
            return std::make_shared<DiagBackendHIP>();
        });
#endif
#endif
    }
    
public:
    // Singleton instance
    static BackendFactory& instance() {
        static BackendFactory factory;
        return factory;
    }
    
    // Delete copy/move constructors
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;
    
    /**
     * Register a backend creator
     */
    void register_backend(const std::string& name, BackendCreator creator) {
        registry_[name] = creator;
    }
    
    /**
     * Create a backend instance
     * Throws if backend not found or not available
     */
    std::shared_ptr<DiagBackend> create(const std::string& backend_name) const {
        auto it = registry_.find(backend_name);
        if (it == registry_.end()) {
            throw std::runtime_error(
                "Backend '" + backend_name + "' not registered. "
                "Available backends: " + available_backends_string()
            );
        }
        
        auto backend = it->second();
        if (!backend->is_available()) {
            throw std::runtime_error(
                "Backend '" + backend_name + "' is registered but not available. "
                "Check hardware and drivers."
            );
        }
        
        return backend;
    }
    
    /**
     * Get list of registered backends
     */
    std::vector<std::string> registered_backends() const {
        std::vector<std::string> backends;
        for (const auto& [name, _] : registry_) {
            backends.push_back(name);
        }
        return backends;
    }
    
    /**
     * Get list of available backends (registered AND available)
     */
    std::vector<std::string> available_backends() const {
        std::vector<std::string> backends;
        for (const auto& [name, creator] : registry_) {
            auto backend = creator();
            if (backend->is_available()) {
                backends.push_back(name);
            }
        }
        return backends;
    }
    
    /**
     * Get default backend (prefer GPU if available)
     */
    std::string default_backend() const {
#ifdef SBD_THRUST
#ifdef __CUDACC__
        try {
            auto cuda = create("cuda");
            if (cuda->is_available()) return "cuda";
        } catch (...) {}
#else
        try {
            auto hip = create("hip");
            if (hip->is_available()) return "hip";
        } catch (...) {}
#endif
#endif
        return "cpu";  // Fallback to CPU
    }
    
private:
    std::string available_backends_string() const {
        auto backends = available_backends();
        std::string result;
        for (size_t i = 0; i < backends.size(); ++i) {
            if (i > 0) result += ", ";
            result += backends[i];
        }
        return result;
    }
};

/**
 * Convenience function to create backend
 * If name is empty, uses default backend
 */
inline std::shared_ptr<DiagBackend> create_backend(const std::string& name = "") {
    auto& factory = BackendFactory::instance();
    if (name.empty()) {
        return factory.create(factory.default_backend());
    }
    return factory.create(name);
}

} // namespace tpb
} // namespace sbd

#endif // SBD_TPB_BACKEND_FACTORY_H
```

### 6. Python Bindings (Single Module, Everything in C++)

```cpp
// File: python/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mpi4py/mpi4py.h>

#include "sbd/tpb/backend_factory.h"

namespace py = pybind11;

MPI_Comm get_mpi_comm(py::object py_comm) {
    PyObject* py_comm_ptr = py_comm.ptr();
    MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm_ptr);
    if (!comm_ptr) throw std::runtime_error("Invalid MPI communicator");
    return *comm_ptr;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "SBD Python bindings with backend support";
    
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Failed to import mpi4py");
    }
    
    // ========================================================================
    // Backend Interface
    // ========================================================================
    
    py::class_<sbd::tpb::DiagBackend, std::shared_ptr<sbd::tpb::DiagBackend>>(
        m, "DiagBackend", "Abstract backend interface")
        .def("name", &sbd::tpb::DiagBackend::name, "Get backend name")
        .def("device_type", &sbd::tpb::DiagBackend::device_type, "Get device type")
        .def("is_available", &sbd::tpb::DiagBackend::is_available, "Check if backend is available");
    
    // ========================================================================
    // Backend Factory Functions
    // ========================================================================
    
    m.def("create_backend",
          [](const std::string& name) {
              return sbd::tpb::create_backend(name);
          },
          py::arg("name") = "",
          "Create a backend instance. Empty string uses default backend.");
    
    m.def("available_backends",
          []() {
              return sbd::tpb::BackendFactory::instance().available_backends();
          },
          "Get list of available backends");
    
    m.def("registered_backends",
          []() {
              return sbd::tpb::BackendFactory::instance().registered_backends();
          },
          "Get list of registered backends (may not all be available)");
    
    m.def("default_backend",
          []() {
              return sbd::tpb::BackendFactory::instance().default_backend();
          },
          "Get default backend name");
    
    // ========================================================================
    // Main Diagonalization Function with Backend Selection
    // ========================================================================
    
    m.def("tpb_diag",
          [](py::object py_comm,
             const sbd::tpb::SBD& sbd_data,
             const sbd::FCIDump& fcidump,
             const std::vector<std::vector<size_t>>& adet,
             const std::vector<std::vector<size_t>>& bdet,
             const std::string& loadname,
             const std::string& savename,
             const std::string& backend_name) {
              
              MPI_Comm comm = get_mpi_comm(py_comm);
              
              // Create backend (all in C++)
              auto backend = sbd::tpb::create_backend(backend_name);
              
              // Prepare output variables
              double energy;
              std::vector<double> density;
              std::vector<std::vector<size_t>> co_adet, co_bdet;
              std::vector<std::vector<double>> one_p_rdm, two_p_rdm;
              
              // Release GIL and call C++ backend
              {
                  py::gil_scoped_release release;
                  backend->diag(comm, sbd_data, fcidump, adet, bdet,
                               loadname, savename, energy, density,
                               co_adet, co_bdet, one_p_rdm, two_p_rdm);
              }
              
              // Return results
              return py::dict(
                  "energy"_a = energy,
                  "density"_a = density,
                  "co_adet"_a = co_adet,
                  "co_bdet"_a = co_bdet,
                  "one_p_rdm"_a = one_p_rdm,
                  "two_p_rdm"_a = two_p_rdm,
                  "backend"_a = backend->name()
              );
          },
          py::arg("comm"),
          py::arg("sbd_data"),
          py::arg("fcidump"),
          py::arg("adet"),
          py::arg("bdet"),
          py::arg("loadname") = "",
          py::arg("savename") = "",
          py::arg("backend") = "",
          "TPB diagonalization with backend selection");
    
    // ... rest of bindings (FCIDump, TPB_SBD, etc.) ...
}
```

### 7. Build System (Conditional Compilation)

```python
# setup.py

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class SBDBuildExt(build_ext):
    """Custom build extension to handle different compilers"""
    
    def build_extensions(self):
        # Detect available backends
        has_cuda = self.check_cuda()
        has_hip = self.check_hip()
        
        for ext in self.extensions:
            # Base flags
            ext.extra_compile_args = ['-std=c++17', '-fopenmp']
            ext.extra_link_args = ['-fopenmp']
            ext.define_macros = []
            
            # Add CUDA support if available
            if has_cuda and os.environ.get('SBD_ENABLE_CUDA', '1') == '1':
                self.configure_cuda(ext)
            # Add HIP support if available
            elif has_hip and os.environ.get('SBD_ENABLE_HIP', '1') == '1':
                self.configure_hip(ext)
            
            # Add MPI
            self.configure_mpi(ext)
        
        super().build_extensions()
    
    def check_cuda(self):
        try:
            subprocess.check_output(['nvcc', '--version'])
            return True
        except:
            return False
    
    def check_hip(self):
        try:
            subprocess.check_output(['hipcc', '--version'])
            return True
        except:
            return False
    
    def configure_cuda(self, ext):
        """Configure CUDA compilation"""
        ext.define_macros.append(('SBD_THRUST', '1'))
        ext.extra_compile_args.extend([
            '-DSBD_THRUST',
            '-I/usr/local/cuda/include',
        ])
        ext.extra_link_args.extend([
            '-L/usr/local/cuda/lib64',
            '-lcudart',
        ])
        print("CUDA support enabled")
    
    def configure_hip(self, ext):
        """Configure HIP compilation"""
        ext.define_macros.append(('SBD_THRUST', '1'))
        ext.extra_compile_args.extend([
            '-DSBD_THRUST',
            '-I/opt/rocm/include',
        ])
        ext.extra_link_args.extend([
            '-L/opt/rocm/lib',
            '-lamdhip64',
        ])
        print("HIP support enabled")
    
    def configure_mpi(self, ext):
        """Configure MPI"""
        try:
            compile_flags = subprocess.check_output(
                ['mpicc', '--showme:compile'],
                universal_newlines=True
            ).strip().split()
            
            link_flags = subprocess.check_output(
                ['mpicc', '--showme:link'],
                universal_newlines=True
            ).strip().split()
            
            for flag in compile_flags:
                if flag.startswith('-I'):
                    ext.include_dirs.append(flag[2:])
            
            for flag in link_flags:
                if flag.startswith('-L'):
                    ext.library_dirs.append(flag[2:])
                elif flag.startswith('-l'):
                    ext.libraries.append(flag[2:])
        except:
            print("Warning: Could not detect MPI flags")

ext_modules = [
    Extension(
        'sbd._core',
        sources=[
            'python/bindings.cpp',
            'sbd/tpb/backend_cpu.cpp',
            # Conditionally compiled:
            # 'sbd/tpb/backend_cuda.cpp',  # Only if CUDA
            # 'sbd/tpb/backend_hip.cpp',   # Only if HIP
        ],
        include_dirs=['.', 'eigen'],
        libraries=['blas', 'lapack'],
        language='c++',
    )
]

setup(
    name='sbd',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': SBDBuildExt},
    # ...
)
```

### 8. Python API Usage

```python
# Example 1: Auto backend selection
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD

# Check what's available
print("Available backends:", sbd.available_backends())
# Output: ['cpu', 'cuda']  (if CUDA is available)

print("Default backend:", sbd.default_backend())
# Output: 'cuda'  (prefers GPU if available)

# Use default backend
results = sbd.tpb_diag(
    comm=comm,
    sbd_data=config,
    fcidump=fcidump,
    adet=adet,
    bdet=bdet
)
print(f"Used backend: {results['backend']}")
```

```python
# Example 2: Explicit backend selection
results_cpu = sbd.tpb_diag(
    comm=comm,
    sbd_data=config,
    fcidump=fcidump,
    adet=adet,
    bdet=bdet,
    backend="cpu"  # Force CPU
)

results_gpu = sbd.tpb_diag(
    comm=comm,
    sbd_data=config,
    fcidump=fcidump,
    adet=adet,
    bdet=bdet,
    backend="cuda"  # Force CUDA
)
```

```python
# Example 3: Backend object
backend = sbd.create_backend("cuda")
print(f"Backend: {backend.name()}")
print(f"Device: {backend.device_type()}")
print(f"Available: {backend.is_available()}")
```

## Key Architectural Decisions

### 1. Everything in C++ Layer
- ✅ Python just passes strings
- ✅ No Python-level backend switching
- ✅ No monkey patching
- ✅ Single compiled module

### 2. Compile-Time Feature Flags
```cpp
#ifdef SBD_THRUST
#ifdef __CUDACC__
    // CUDA code
#else
    // HIP code
#endif
#endif
```

### 3. Dynamic Linking
- Link to CUDA/HIP libraries at build time
- Link to MPI library at build time
- Single binary with all backends

### 4. Factory Registry
- Backends registered at static initialization
- Conditional registration based on compile flags
- Runtime availability checking

### 5. C++ Polymorphism
- Abstract base class `DiagBackend`
- Virtual methods for backend operations
- Factory returns `shared_ptr<DiagBackend>`

## Benefits

1. **Single Build**: One `pip install` gets all available backends
2. **Runtime Selection**: Switch backends without recompilation
3. **Clean API**: Python code is simple, C++ does the work
4. **Type Safety**: C++ polymorphism ensures correctness
5. **Performance**: No Python overhead, direct C++ calls
6. **Extensibility**: Easy to add new backends

## Implementation Checklist

- [ ] Create `backend_interface.h` with abstract `DiagBackend` class
- [ ] Implement `DiagBackendCPU` in `backend_cpu.{h,cpp}`
- [ ] Implement `DiagBackendCUDA` in `backend_cuda.{h,cpp}` (conditional)
- [ ] Implement `DiagBackendHIP` in `backend_hip.{h,cpp}` (conditional)
- [ ] Create `BackendFactory` in `backend_factory.h`
- [ ] Update Python bindings to use factory
- [ ] Update `setup.py` for conditional compilation
- [ ] Add tests for each backend
- [ ] Update documentation

## Summary

This architecture follows PyTorch's proven design:
- **C++ polymorphism** for backend abstraction
- **Factory + registry** for backend creation
- **Compile-time flags** for conditional compilation
- **Dynamic linking** to backend libraries
- **Python passes strings**, C++ does everything
- **No runtime patching**, everything resolved in C++
- **Single module**, all backends in one binary

The result is a clean, efficient, and extensible backend system that provides excellent user experience while maintaining high performance.