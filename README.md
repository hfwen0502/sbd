# Library for selected basis diagonalization

This is a header-only library for diagonalizing quantum systems in a selected basis, with a focus on handling wavefunction vectors that are too large to fit in the memory of a single node.
The library leverages MPI-based parallelization to distribute the wavefunction across multiple nodes.
Sample usage examples are provided in the `/samples` directory.

## Author

- Tomonori Shirakawa, RIKEN Center for Computational Science

## Versions

- **v1.0.0**: Initial public release corresponding to the arXiv submission.
- **v1.1.0**: Feature additions, refactoring, and bug fixes

## Requirement

- Message Passing Interface (MPI)
- OpenMP
- BLAS and LAPACK

## Install

- This code is provided as a header-only llibrary, so no installation is required.

## How to Compile the Sample Codes

- The sample code for parallelized selected basis diagonalization is located in `sample/selected_basis_diagonalization`.
- Edit the configuration file to suit your environment and build it with the make command.
- For more information and options for the executable, see README.md in the same directory.

## Documentation

For more details on the input file formats and internal structure, see the [User Manual](https://www.doxygen.nl/manual/doxygen_usage.html).
You can generate the documentation by running:
```
doxygen ./doc/Doxyfile
```

## Note
This repository contains research code related to the following paper:

- **Title:** Closed-loop calculations of electronic structure on a quantum processor and a classical supercomputer at full scale
- **arXiv:** https://arxiv.org/abs/2511.00224

Version **v1.0.0** corresponds to the code used for the above arXiv submission and represents the initial public release associated with that paper.

Subsequent versions (v1.1.0 and later) include additional features, refactoring, and bug fixes, and may go beyond the exact implementation described in the paper.

The code is shared publicly to support transparency and academic collaboration.  
If you use this repository in your research, please cite the corresponding arXiv paper.

## Licence

[Apache License 2.0](https://github.com/r-ccs-cms/sbd/blob/main/LICENSE.txt)
