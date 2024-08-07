# Approximate Sparse Linear Algebra

Bayesian Gibbs sampler for GLMMs with Polya Gamma augmentation. 

The function `PGgibbs_GLMMs(df::DataFrame, f::FormulaTerm, n_iter::Integer; kwargs...)` returns MCMC posterior samples.

The algorithms in this package are presented in:

[Andrea Pandolfi, Omiros Papaspiliopoulos, and Giacomo Zanella. "Conjugate Gradient Method for
High-dimensional GLMMs."](link)

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/andreapandolfi/ASLA.jl")
```



## Usage

### Exact inference

```julia
using ASLA, MixedModels, DataFrames

df = DataFrame(MixedModels.dataset(:verbagg))
f = @formula(r2 ~ 1 + anger + gender + situ + (1|subj) + (1|item) + (1|subj&item));

n_iters = 1000; burn_in = 200;
β, T = PGgibbs_GLMMs(df, f, n_iters);
```

### Approximate inference

```julia
using IterativeSolvers
β_cg, T_cg = PGgibbs_GLMMs(df, f, n_iters, burn_in=burn_in, system_solver! = (x, Q, b) -> cg!(x,Q, b, Pl=JacobiPreconditioner(Q)));
```
