using ASLA
using MixedModels, DataFrames

df = DataFrame(MixedModels.dataset(:verbagg))
f = @formula(r2 ~ 1 + anger + gender + situ + (1|subj) + (1|item) + (1|subj&item));

n_iters = 50; burn_in = 20;
β, T = PGgibbs_GLMMs(df, f, n_iters);

using IterativeSolvers
β_cg, T_cg = PGgibbs_GLMMs(df, f, n_iters, burn_in=burn_in, system_solver! = (x, Q, b) -> cg!(x,Q, b, Pl=JacobiPreconditioner(Q)));