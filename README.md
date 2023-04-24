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

```bash
cd applications/BENCHMARK_CLAERBOUT_ADJOINT
./helper_tyler.py --help
```

In the `helper_tyler.py` script, you will see main different options that can be run. The main two options are (a) gradient-descent based inversion for $L^2$,
$W_2$, or $H^s$ misfits, and (b) generation of full landscapes based on source location (two-layer velocity models planned for future). Option (b) is for 
pedagogical purposes to study how difficult the underlying optimization problem is for comparison between misfits.
