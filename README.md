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
pedagogical purposes to study how difficult the underlying optimization problem is for comparison between misfits. Here is a table that describes the options 
that would be seen when running `./helper_tyler.py --help`

| Option           | Type    | Default       | Description                                             |
|------------------|---------|---------------|---------------------------------------------------------|
| mode             | int     | (required)    | Mode of execution (int: 1 <= mode <= 8)                 |
| --misfit         | str     | "l2"          | Misfit functional, either l2 or w2                      |
| --plot           | flag    | False         | Perform seismogram plots                                |
| --num_sources    | int     | 10            | Convexity plot granularity                              |
| --rerun          | flag    | False         | Rerun forward solving                                   |
| --recompute      | flag    | False         | Recompute misfit                                        |
| --restrict       | float   | None          | Wasserstein restriction                                 |
| --ext            | str     | "su"          | Data file extension                                     |
| --log            | str     | "logger.log"  | Logger file                                             |
| --store_all      | flag    | False         | Store all files                                         |
| --max_proc       | int     | 16            | Max number of spawned processes at one time             |
| --purge_interval | int     | 100           | Every purge_interval steps, extra files eliminated      |
| --s              | float   | 0.0           | Sobolev smoothing parameter                             |
| --stride         | int     | 1             | Downsampling parameter for landscape plots              |
| --suffix         | str     | "0"           | Appending to output file                                |

