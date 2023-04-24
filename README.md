# SpecFEMInversion

Welcome to the **SpecFEMInversion**! 

This repository currently contains an implementation of elastic moment source inversion for $L^2$ and $W_2$ misfits.
We plan to incorporate velocity inversion and joint source-velocity inversion going forward.

If you have interest in adding features, please add a pull request or contact me directly at tyler@oden.utexas.edu.

If you have any trouble with our code or find bugs, please let us know by filling out an issue!

## Table of Contents

- [Getting Started](#getting-started)
- [How to Use this Software](#how-to-use)

## Getting Started

The main depenency is the SPECFEM2D project whose installation instructions can be found in the README for the GitHub project found 
[here](https://github.com/SPECFEM/specfem2d).

After SPECFEM2D is installed, clone this repo with the following commands:

```bash
git clone https://github.com/tmasthay/SpecFEMInversion.git
cd SpecFEMInversion
```

## How to Use 

(1) One-dimensional $W_1$ and $W_2$ (trace-by-trace) 
  - [Yang et al. 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Application+of+optimal+transport+and+the+quadratic+Wasserstein+metric+to+full-waveform+inversion&btnG=)

(2) Huber
  - [Guitton and Symes 2003](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Guitton%2C+A.%2C+and+W.+W.+Symes%2C+2003%2C+Robust+inversion+of+seismic+data+using+the+Huber+norm%3A+Geophysics&btnG=)

(3) $L^1$-$L^2$ hybrid
  - [Bube and Langan](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Hybrid+l1%E2%88%95l2+minimization+with+applications+to+tomography&btnG=)

(4) Normalized and unnormalized Sobolev norms

  - [Zhu et al. 2021](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Implicit+regularization+effects+of+the+Sobolev+norms+in+image+processing&btnG=)

## Misfit Functions (roadmap for future)

(1) Fisher-Rao metric

  - [Zhou et al. 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=The+Wasserstein-Fisher-Rao+metric+for+waveform+based+earthquake+location&btnG=)

(2) Graph-space OT

  - [Metivier et al. 2018](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Optimal+transport+for+mitigating+cycle+skipping+in+full-waveform+inversion%3A+A+graph-space+transform+approach&btnG=)

(3) Entropic regularization OT

  - [Cuturi 2013](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Sinkhorn+distances%3A+Lightspeed+computation+of+optimal+transport&btnG=)

(4) Misfits based on reduced-order models

  - [Borcea et al. 2023](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C44&q=Waveform+inversion+via+reduced+order+modeling+borcea&btnG=)
