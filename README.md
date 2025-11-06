# GROMACS-ESP

GROMACS-ESP is a fork of [GROMACS](https://github.com/gromacs/gromacs) that preserves the GROMACS user 
interface while replacing the treatment of Coulomb interactions with ESP—an Ewald-summation variant 
based on prolate spheroidal wave functions (PSWFs). 

# Background

Molecular dynamics (MD) simulations are essential tools in materials science, computational chemistry, structural biology, and drug discovery. Fast Ewald summation is
the most widely used method for evaluating long-range Coulomb interactions in MD, and it
remains a performance bottleneck in all major open-source and commercial MD packages,
particularly in massively parallel simulations involving $10^9 - 10^{12}$ time steps. Notably, LAMMPS ($>80,000$ citations on Google Scholar), 
[GROMACS](https://scholar.google.nl/citations?user=IHbqqNEAAAAJ&hl=nl)($>85,000$ citations), and NAMD
all rely on Ewald-based approaches, such as the (Smooth) Particle-Mesh Ewald ([PME/SPME](https://scholar.google.com/citations?user=y4Ts-LkAAAAJ&hl=en), $>57,000$ citations)
and the Particle-Particle-Particle-Mesh (PPPM, $>12,000$ citations) methods. 

# Introduction

Without any loss of accuracy, ESP alters the fast Ewald pipeline in two places. 
First, for kernel splitting it uses PSWFs instead of Gaussians, which—thanks to the optimal concentration 
of PSWFs among band-limited functions—significantly reduces the required Fourier grid. With everything else 
equal, the FFT length drops by about a factor of two per dimension at high accuracy (≈8× in 3D). The 
residual kernel also vanishes at the real-space cutoff, eliminating any need for an “energy shift.” 
Second, for particle–mesh operations ESP employs PSWFs in place of the B-splines used by SPME. For comparable 
accuracy without k-space upsampling, PSWFs require fewer neighboring grid points (e.g., ~8 vs ~12 for five-digit accuracy). 
In contrast, native GROMACS typically sets the spreading/interpolation order to P=5 for MPI runs and P=4 for 
single-core low-accuracy runs, which forces substantial Fourier-space upsampling. Consequently, at the same cutoff 
radius, native GROMACS often needs a much larger FFT grid—even for ≈10⁻³ force accuracy—whereas ESP achieves similar 
accuracy with far shorter transforms, yielding roughly a sixfold reduction in FFT length when other parameters are held fixed.
We implemented ESP as modular components in both GROMACS and LAMMPS, introducing only minimal, localized changes to enable 
rigorous and fair comparisons with the native codes.

## Problem formulation
Consider a charged system of $N$ particles located at $`\mathbf{r}_{i}`$, $i=1,\cdots,N$, with
charge strengths $q_{i}$, in the periodic domain $\Omega=[-L/2,L/2]^3$;
we present the cubic case for simplicity.
A core calculation in MD simulations is the evaluation of 
the electrostatic potentials $`u(\mathbf{r}_i)`$ at these locations, 
excluding the self-contribution:

$$
u_i := u(\mathbf{r}_i)=
\sum_{\substack{j\in\{1,\dots,N\},\ \mathbf{n}\in\mathbb{Z}^3 \\\ (j,\mathbf{n})\neq(i,\mathbf{0})}}
\frac{q_j}{4\pi\,\bigl\lvert \mathbf{r}_{i}-\mathbf{r}_j-\mathbf{n}L\bigr\rvert},
\qquad i=1,\dots,N.
$$

The total electrostatic energy is $E=\frac{1}{2}\sum_{i=1}^{N}q_i\,u_i$, and the electrostatic force acting on the $`i`$ th particle is
$`\mathbf{F}(\mathbf{r}_i)=-\nabla_{\mathbf{r}_i}E`$.
Here we have nondimensionalized
so that physical constants such as $\epsilon_0$ do not appear.
For periodic systems, it is well known that 
the system must satisfy the charge neutrality condition $`\sum_{j=1}^{N}q_j=0`$.
The potential $u$ is only determined up to an arbitrary additive constant.

## Algorithm at a glance
- Real-space (short range): evaluate damped pairwise interactions using a cutoff.
- Reciprocal-space (long range):
  - Spread charges to a uniform grid with a separable PSWF window of width $`P`$.
  - Apply a smaller 3D FFT (fewer grid points for the same accuracy).
  - Multiply the fourier coefficients by the PSWF-based influence function.
  - Inverse FFT to return to real space.
  - Gather potentials/forces at particle locations using the same PSWF window.
- Add self corrections as required by the chosen split.

Rather than the traditional Ewald split, the ESP method
considers a general radially symmetric split of the Coulomb kernel into
spectral (smooth long-range) plus local (short-range) parts:

$$
\frac{1}{4\pi r}=S(r) + L(r)
:= \frac{\displaystyle\int_0^{r/r_c}\chi_\alpha(x)\,dx}{4\pi r} + 
\frac{1-\displaystyle\int_0^{r/r_c}\chi_\alpha(x)\,dx}{4\pi r},
$$

where $r_c$ defines a cutoff (truncation) radius,
and $\alpha>0$ is a “shape parameter” depending on the desired precision $\varepsilon$.
Here $\chi_\alpha$ is a smooth, nonnegative, even function with normalization
$\int_0^1 \chi_\alpha(x)\,dx = 1 - \mathcal{O}(\varepsilon)$,
and $\varepsilon$-support of $[-1,1]$, meaning $\chi_\alpha(1) \approx \varepsilon$
and $\chi_\alpha(x)$ decays rapidly to zero as $x\rightarrow \infty$.
The short-range component $L(r)$
may be truncated at $r=r_c$ to precision $\varepsilon$.
This allows the corresponding “local” part of the potential

$$
u_i^{\ell} =
\sum_{\substack{j\in\{1,\dots,N\},\ \mathbf{n}\in\mathbb{Z}^3 \\\ (j,\mathbf{n})\neq(i,\mathbf{0})}}
L\left(\left\lvert \mathbf{r}_{i}-\mathbf{r}_j-L\mathbf{n}\right\rvert\right)\, q_j
$$

to be computed directly in $O(Ns)$ cost, where $s$ is the average number of particles
within distance $r_c$ of a particle.

# Download Instructions

This repository is self-contained with no external submodules, so a standard `git clone` is sufficient.

```bash
git clone https://github.com/lu1and10/Ewald-Splitting-with-Prolates.git
```

# Installation Guide

Installation is identical to native GROMACS. Follow the official GROMACS installation guide for build and configuration details:  
<https://manual.gromacs.org/documentation/current/install-guide/index.html>

# User Guide

The user interface is identical to native GROMACS, so you can rely on the official GROMACS manual for general 
usage—including the molecular topology file (`.top`), molecular dynamics parameter file (`.mdp`), run input file (`.tpr`), 
and related workflows:  
<https://manual.gromacs.org/current/user-guide/index.html>

This fork changes only the algorithms used for Coulomb interaction calculations. The `.mdp` options relevant to Coulomb 
interactions are the only user-visible differences and are discussed below.

# Changes to the `.mdp` file

The native GROMACS `.mdp` file exposes many parameters that most users neither need nor benefit from tuning. In ESP, 
the algorithmic choices that matter boil down to **two** quantities:

- the real-space cutoff radius $r_c$, and  
- the target relative **force** error  $\epsilon$.

For Coulomb interactions, $r_c$ splits work between the near-field (evaluated directly with the residual kernel) 
and the long-range part (handled by the fast Ewald method).

## Practical note on the cutoff
Most MD setups also include Lennard–Jones (van der Waals) interactions that are sharply truncated at the same cutoff. 
Using a very small $r_c$ can therefore introduce large errors for Lennard–Jones, independent of the Coulomb treatment. 
**We recommend keeping your current cutoff** (as in native GROMACS) for now. We are implementing ESP for the 
Lennard–Jones potential; once available, you will be able to reduce $r_c$ to accelerate simulations **without** sacrificing accuracy.

## Parameters to set

- **`rcoulomb`** (the cutoff $r_c$):  
  Keep this equal to your usual production value to avoid Lennard–Jones truncation artifacts until ESP–LJ is released.

- **`ewald_rtol`** (target tolerance $\epsilon$):  
  In native GROMACS this is commonly interpreted as a tolerance for **potential** accuracy and is often set to `1e-5`.  
  **In this fork, `ewald_rtol` specifies the desired *force* accuracy.** Typical choices are:
  - `1e-3` to `1e-4` for most MD simulations,
  - smaller values only if your application demands tighter force accuracy.

- **`pme_order`** (PSWF spreading/interpolation order):  
  This controls the order of the prolate spheroidal basis used for particle–mesh operations.
  - `pme_order = 4` → suitable for ~3 digits of accuracy,  
  - `pme_order = 5` → suitable for ~4 digits of accuracy.  
  Values above 5 are rarely necessary.

## Parameters you do **not** need to set

You do **not** need to adjust PME/Fourier grid controls (e.g., Fourier spacing or FFT grid sizes). 
ESP computes the required k-space parameters internally to meet your `ewald_rtol` target at the chosen $r_c$.

In summary: set a sensible `rcoulomb`, choose `ewald_rtol` for your **force** accuracy needs (typically `1e-3`–`1e-4`), 
pick `pme_order` = 4 or 5, and leave the rest to ESP’s internal autotuning.

## Overall accuracy of MD simulations

The overall accuracy of molecular dynamics (MD) simulations is influenced by several sources of error: modeling (force-field and parameterization), system preparation and input files, and numerical error from the time integrator and related algorithms. Even though our implementation ensures that forces are evaluated to the requested tolerance at each time step, the fidelity of the trajectory is ultimately constrained by these factors and by other components of the original GROMACS code base (e.g., integrators, constraints, thermostats). Note that the potential/energy is typically at least ten times more accurate than the force field.

# Citing

If GROMACS-ESP is useful in your work, please star this repository and cite both the software and the reference below.

### Preferred citation (BibTeX)
```bibtex
@misc{liang2025arxiv,
  title         = {Accelerating Fast Ewald Summation with Prolates for Molecular Dynamics Simulations},
  author        = {Jiuyang Liang and Libin Lu and Alex Barnett and Leslie Greengard and Shidong Jiang},
  year          = {2025},
  eprint        = {2505.09727},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  url           = {https://arxiv.org/abs/2505.09727}
}
```

# Main developers

- **Libin Lu** — Lead developer  
  *Flatiron Institute, Simons Foundation*
- **Jiuyang Liang**  
  *Flatiron Institute, Simons Foundation*
- **Alex Barnett**  
  *Flatiron Institute, Simons Foundation*
- **Leslie Greengard**  
  *Flatiron Institute, Simons Foundation*
- **Shidong Jiang**  
  *Flatiron Institute, Simons Foundation*


# Availability & Contact

The datasets and input files for testing the PME and ESP methods can be found in the gromacs_bench directory.
More detailed documentation will be available soon. If you have questions or feedback,
please email **Libin Lu** at <llu@flatironinstitute.org>.

**Disclaimer:** This codebase is **not** an official release of GROMACS. It is independently 
maintained and modified. 



                               * * * * *

GROMACS is free software, distributed under the GNU Lesser General
Public License, version 2.1 However, scientific software is a little
special compared to most other programs. Both you, we, and all other
GROMACS users depend on the quality of the code, and when we find bugs
(every piece of software has them) it is crucial that we can correct
it and say that it was fixed in version X of the file or package
release. For the same reason, it is important that you can reproduce
other people's result from a certain GROMACS version.

The easiest way to avoid this kind of problems is to get your modifications
included in the main distribution. We'll be happy to consider any decent 
code. If it's a separate program it can probably be included in the contrib 
directory straight away (not supported by us), but for major changes in the 
main code we appreciate if you first test that it works with (and without) 
MPI, threads, double precision, etc.

If you still want to distribute a modified version or use part of GROMACS
in your own program, remember that the entire project must be licensed
according to the requirements of the LGPL v2.1 license under which you
received this copy of GROMACS. We request that it must clearly be labeled as
derived work. It should not use the name "official GROMACS", and make
sure support questions are directed to you instead of the GROMACS developers.
Sorry for the hard wording, but it is meant to protect YOUR research results!

                               * * * * *

