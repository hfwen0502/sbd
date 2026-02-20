/**
 * @file python/bindings.cpp
 * @brief Python bindings for SBD TPB diagonalization using pybind11
 *
 * This file is compiled twice with different module names:
 * - _core_cpu: CPU backend
 * - _core_gpu: GPU backend (with -DSBD_THRUST)
 *
 * The module name is controlled by the SBD_MODULE_NAME macro.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include "sbd/sbd.h"
#include "sbd/chemistry/basic/csr_export.h"

namespace py = pybind11;

/**
 * Helper function to convert mpi4py communicator to MPI_Comm
 */
MPI_Comm get_mpi_comm(py::object py_comm) {
    PyObject* py_comm_ptr = py_comm.ptr();
    MPI_Comm* comm_ptr = PyMPIComm_Get(py_comm_ptr);
    if (!comm_ptr) {
        throw std::runtime_error("Invalid MPI communicator");
    }
    return *comm_ptr;
}

// Module name is set by compiler flag -DSBD_MODULE_NAME=_core_cpu or _core_gpu
#ifndef SBD_MODULE_NAME
#define SBD_MODULE_NAME _core
#endif

PYBIND11_MODULE(SBD_MODULE_NAME, m) {
    // Set module docstring based on backend
#ifdef SBD_THRUST
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library - GPU backend";
#else
    m.doc() = "Python bindings for SBD (Selected Basis Diagonalization) library - CPU backend";
#endif

    // Initialize mpi4py
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Failed to import mpi4py");
    }

    // ========================================================================
    // Bind FCIDump structure
    // ========================================================================
    py::class_<sbd::FCIDump>(m, "FCIDump", "FCIDUMP data structure")
        .def(py::init<>())
        .def_readwrite("header", &sbd::FCIDump::header,
                      "Header information as dictionary (map<string, string>)")
        .def_readwrite("integrals", &sbd::FCIDump::integrals,
                      "Integral data as list of tuples (value, i, j, k, l)");

    // ========================================================================
    // Bind TPB SBD configuration structure
    // ========================================================================
    py::class_<sbd::tpb::SBD>(m, "TPB_SBD", "Configuration for TPB diagonalization")
        .def(py::init<>())
        .def_readwrite("task_comm_size", &sbd::tpb::SBD::task_comm_size,
                      "Task communicator size")
        .def_readwrite("adet_comm_size", &sbd::tpb::SBD::adet_comm_size,
                      "Alpha determinant communicator size")
        .def_readwrite("bdet_comm_size", &sbd::tpb::SBD::bdet_comm_size,
                      "Beta determinant communicator size")
        .def_readwrite("h_comm_size", &sbd::tpb::SBD::h_comm_size,
                      "Helper communicator size")
        .def_readwrite("method", &sbd::tpb::SBD::method,
                      "Diagonalization method (0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham)")
        .def_readwrite("max_it", &sbd::tpb::SBD::max_it,
                      "Maximum number of iterations")
        .def_readwrite("max_nb", &sbd::tpb::SBD::max_nb,
                      "Maximum number of basis vectors")
        .def_readwrite("eps", &sbd::tpb::SBD::eps,
                      "Convergence tolerance")
        .def_readwrite("max_time", &sbd::tpb::SBD::max_time,
                      "Maximum time in seconds")
        .def_readwrite("init", &sbd::tpb::SBD::init,
                      "Initialization method")
        .def_readwrite("do_shuffle", &sbd::tpb::SBD::do_shuffle,
                      "Shuffle determinants flag")
        .def_readwrite("do_rdm", &sbd::tpb::SBD::do_rdm,
                      "Calculate RDM flag (0=density only, 1=full RDM)")
        .def_readwrite("carryover_type", &sbd::tpb::SBD::carryover_type,
                      "Carryover determinant selection type")
        .def_readwrite("ratio", &sbd::tpb::SBD::ratio,
                      "Carryover ratio")
        .def_readwrite("threshold", &sbd::tpb::SBD::threshold,
                      "Carryover threshold")
        .def_readwrite("bit_length", &sbd::tpb::SBD::bit_length,
                      "Bit length for determinant representation")
        .def_readwrite("dump_matrix_form_wf", &sbd::tpb::SBD::dump_matrix_form_wf,
                      "Filename to dump wavefunction in matrix form")
#ifdef SBD_THRUST
        .def_readwrite("use_precalculated_dets", &sbd::tpb::SBD::use_precalculated_dets,
                      "Use precalculated determinants (THRUST)")
        .def_readwrite("max_memory_gb_for_determinants", &sbd::tpb::SBD::max_memory_gb_for_determinants,
                      "Maximum memory in GB for determinants (THRUST)")
#endif
        ;

    // ========================================================================
    // Utility functions
    // ========================================================================
    
    m.def("LoadFCIDump", &sbd::LoadFCIDump,
          "Load FCIDUMP file and return FCIDump object",
          py::arg("filename"));

    m.def("LoadAlphaDets",
          [](const std::string& filename, size_t bit_length, size_t total_bit_length) {
              std::vector<std::vector<size_t>> dets;
              sbd::LoadAlphaDets(filename, dets, bit_length, total_bit_length);
              return dets;
          },
          "Load alpha determinants from file",
          py::arg("filename"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    m.def("makestring", &sbd::makestring,
          "Convert bitstring to string representation",
          py::arg("config"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    m.def("from_string", &sbd::from_string,
          "Convert binary string to determinant format",
          py::arg("s"),
          py::arg("bit_length"),
          py::arg("total_bit_length"));

    // ========================================================================
    // Main TPB diagonalization function (data structure version)
    // ========================================================================
    
    m.def("tpb_diag",
        [](py::object py_comm,
           const sbd::tpb::SBD& sbd_data,
           const sbd::FCIDump& fcidump,
           const std::vector<std::vector<size_t>>& adet,
           const std::vector<std::vector<size_t>>& bdet,
           const std::string& loadname,
           const std::string& savename) {
            
            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);
            
            // Get MPI rank for GPU assignment
            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);
            
#ifdef SBD_THRUST
            // Assign GPU device based on MPI rank
            int numDevices, myDevice;
#ifdef __CUDACC__
            cudaGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            cudaSetDevice(myDevice);
#else
            hipGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            hipSetDevice(myDevice);
#endif
#endif
            
            // Output variables
            double energy;
            std::vector<double> density;
            std::vector<std::vector<size_t>> co_adet;
            std::vector<std::vector<size_t>> co_bdet;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;
            
            // Release GIL for long computation
            py::gil_scoped_release release;
            
            // Call C++ function
            sbd::tpb::diag(comm, sbd_data, fcidump, adet, bdet,
                          loadname, savename, energy, density,
                          co_adet, co_bdet, one_p_rdm, two_p_rdm);
            
            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;
            
            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_adet"] = co_adet;
            results["carryover_bdet"] = co_bdet;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            
            return results;
        },
        "Perform TPB diagonalization with pre-loaded data structures",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidump"),
        py::arg("adet"),
        py::arg("bdet"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

    // ========================================================================
    // Main TPB diagonalization function (file-based version)
    // ========================================================================
    
    m.def("tpb_diag_from_files",
        [](py::object py_comm,
           const sbd::tpb::SBD& sbd_data,
           const std::string& fcidumpfile,
           const std::string& adetfile,
           const std::string& loadname,
           const std::string& savename) {
            
            // Convert MPI communicator
            MPI_Comm comm = get_mpi_comm(py_comm);
            
            // Get MPI rank for GPU assignment
            int mpi_rank;
            MPI_Comm_rank(comm, &mpi_rank);
            
#ifdef SBD_THRUST
            // Assign GPU device based on MPI rank
            int numDevices, myDevice;
#ifdef __CUDACC__
            cudaGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            cudaSetDevice(myDevice);
#else
            hipGetDeviceCount(&numDevices);
            myDevice = mpi_rank % numDevices;
            hipSetDevice(myDevice);
#endif
#endif
            
            // Output variables
            double energy;
            std::vector<double> density;
            std::vector<std::vector<size_t>> co_adet;
            std::vector<std::vector<size_t>> co_bdet;
            std::vector<std::vector<double>> one_p_rdm;
            std::vector<std::vector<double>> two_p_rdm;
            
            // Release GIL for long computation
            py::gil_scoped_release release;
            
            // Call file-based C++ function
            sbd::tpb::diag(comm, sbd_data, fcidumpfile, adetfile,
                          loadname, savename, energy, density,
                          co_adet, co_bdet, one_p_rdm, two_p_rdm);
            
            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;
            
            // Return results as dictionary
            py::dict results;
            results["energy"] = energy;
            results["density"] = density;
            results["carryover_adet"] = co_adet;
            results["carryover_bdet"] = co_bdet;
            results["one_p_rdm"] = one_p_rdm;
            results["two_p_rdm"] = two_p_rdm;
            
            return results;
        },
        "Perform TPB diagonalization from files (convenience function)",
        py::arg("comm"),
        py::arg("sbd_data"),
        py::arg("fcidumpfile"),
        py::arg("adetfile"),
        py::arg("loadname") = "",
        py::arg("savename") = "");

    // ========================================================================
    // CSR Hamiltonian Export
    // ========================================================================
    
    m.def("export_hamiltonian_csr",
        [](const sbd::FCIDump& fcidump,
           const std::vector<std::vector<size_t>>& adet,
           const std::vector<std::vector<size_t>>& bdet,
           size_t bit_length,
           size_t max_nnz) {
            
            size_t n_adet = adet.size();
            size_t n_bdet = bdet.size();
            size_t n = n_adet * n_bdet;  // Total Hilbert space dimension
            
            if (n > 100000) {
                throw std::runtime_error(
                    "Matrix dimension too large for CSR export.\n"
                    "Current limit: 100,000 x 100,000\n"
                    "For larger problems, use SBD's built-in Davidson solver."
                );
            }
            
            // Extract integrals from FCIDump using SetupIntegrals
            int L = 0;  // number of orbitals
            int N = 0;  // number of electrons
            double I0 = 0.0;  // core energy
            sbd::oneInt<double> I1;
            sbd::twoInt<double> I2;
            
            sbd::SetupIntegrals(fcidump, L, N, I0, I1, I2);
            size_t norb = static_cast<size_t>(L);
            
            // Verify integrals were properly initialized
            if (L == 0 || I1.store.empty() || I2.store.empty()) {
                throw std::runtime_error(
                    "Failed to initialize integrals from FCIDUMP.\n"
                    "Check that FCIDUMP file contains valid NORB and integral data."
                );
            }
            
            // Additional validation
            if (I2.DirectMat.empty() || I2.ExchangeMat.empty()) {
                throw std::runtime_error(
                    "Failed to initialize Direct/Exchange matrices.\n"
                    "I2.DirectMat size: " + std::to_string(I2.DirectMat.size()) +
                    ", I2.ExchangeMat size: " + std::to_string(I2.ExchangeMat.size()) +
                    ", Expected: " + std::to_string(L * L)
                );
            }
            
            // Debug info
            if (py::module_::import("sys").attr("stdout").attr("isatty")().cast<bool>()) {
                py::print("Debug: L =", L, ", N =", N, ", I0 =", I0);
                py::print("Debug: I1.norbs =", I1.norbs, ", I1.store.size() =", I1.store.size());
                py::print("Debug: I2.norbs =", I2.norbs, ", I2.store.size() =", I2.store.size());
                py::print("Debug: I2.DirectMat.size() =", I2.DirectMat.size());
                py::print("Debug: I2.ExchangeMat.size() =", I2.ExchangeMat.size());
            }
            
            // Build Hamiltonian in triplet format
            std::vector<sbd::MatrixTriplet<double>> triplets;
            
            // Release GIL for long computation
            py::gil_scoped_release release;
            
            bool completed = sbd::buildHamiltonianTriplets(
                adet, bdet, bit_length, norb, I0, I1, I2, max_nnz, triplets
            );
            
            // Convert to CSR format
            std::vector<double> data;
            std::vector<int> indices;
            std::vector<int> indptr;
            
            sbd::tripletsToCSR(triplets, n, data, indices, indptr);
            
            // Reacquire GIL for Python object creation
            py::gil_scoped_acquire acquire;
            
            // Convert to NumPy arrays
            py::array_t<double> np_data(data.size(), data.data());
            py::array_t<int> np_indices(indices.size(), indices.data());
            py::array_t<int> np_indptr(indptr.size(), indptr.data());
            
            // Return results as dictionary
            py::dict result;
            result["data"] = np_data;
            result["indices"] = np_indices;
            result["indptr"] = np_indptr;
            result["shape"] = py::make_tuple(n, n);
            result["nnz"] = data.size();
            result["truncated"] = !completed;
            
            return result;
        },
        py::arg("fcidump"),
        py::arg("adet"),
        py::arg("bdet"),
        py::arg("bit_length"),
        py::arg("max_nnz") = 100000000,
        R"pbdoc(
            Export Hamiltonian matrix in CSR (Compressed Sparse Row) format.
            
            WARNING: Not yet fully implemented. This is a placeholder for future development.
            
            Args:
                fcidump: FCIDump object with molecular integrals
                adet: Alpha determinants
                bdet: Beta determinants
                bit_length: Bit length for determinant representation
                max_nnz: Maximum number of non-zero elements (default: 10^8)
            
            Returns:
                dict: CSR format data (when implemented)
                    - 'data': Non-zero values
                    - 'indices': Column indices
                    - 'indptr': Row pointers
                    - 'shape': Matrix dimensions
                    - 'nnz': Number of non-zeros
                    - 'truncated': Whether matrix was truncated
            
            Raises:
                RuntimeError: Feature not yet implemented
        )pbdoc");

    // ========================================================================
    // Cleanup/Finalization functions
    // ========================================================================
    
    m.def("cleanup_device",
        []() {
#ifdef SBD_THRUST
            // Synchronize GPU device but do NOT reset
            // cudaDeviceReset() can interfere with CUDA-aware MPI (UCX)
            // which may still have active CUDA events/streams
#ifdef __CUDACC__
            cudaDeviceSynchronize();
            // Note: cudaDeviceReset() intentionally NOT called to avoid
            // conflicts with CUDA-aware MPI cleanup
#else
            hipDeviceSynchronize();
            // Note: hipDeviceReset() intentionally NOT called to avoid
            // conflicts with ROCm-aware MPI cleanup
#endif
#endif
        },
        "Synchronize GPU device (GPU backend only). "
        "Note: Does not call cudaDeviceReset() to avoid conflicts with CUDA-aware MPI. "
        "GPU resources are freed automatically when the process exits.");

    m.def("finalize_mpi",
        []() {
            // Check if MPI is initialized before finalizing
            int initialized, finalized;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            
            if (initialized && !finalized) {
                MPI_Finalize();
            }
        },
        "Finalize MPI. Only call this if you initialized MPI yourself. "
        "If using mpi4py, MPI finalization is handled automatically at exit.");

    // ========================================================================
    // Version information
    // ========================================================================
    
    m.attr("__version__") = "1.2.0";
}

// Made with Bob
