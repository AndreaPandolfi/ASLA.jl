using Test
using LinearAlgebra, IterativeSolvers
using ASLA: JacobiPreconditioner, ICholPreconditioner, W2_sample_distance

@testset "Approximate CG Gibbs" begin
    verbagg = MixedModels.dataset(:verbagg)
    f = @formula(r2 ~ 1 + anger + gender + btype + situ + (1|subj) + (1|item) + (1|subj&item));
    df = DataFrame(verbagg)
    n_iters = 50; burn_in = 20; seed = 18;
    β, T                = PGgibbs_GLMMs(df, f, n_iters, burn_in=burn_in, seed=seed);
    β_jac, T_jac        = PGgibbs_GLMMs(df, f, n_iters, burn_in=burn_in, seed=seed, system_solver! = (x, Q, b) -> cg!(x,Q, b, Pl=JacobiPreconditioner(Q)))
    β_ichol, T_ichol    = PGgibbs_GLMMs(df, f, n_iters, burn_in=burn_in, seed=seed, system_solver! = (x, Q, b) -> cg!(x,Q, b, Pl=ICholPreconditioner(Q)))

    @test length(β)==length(T)
    @test length(β[1])== length(β_jac[1])
    @test length(β[1])== length(β_ichol[1])

    W2_β = W2_sample_distance(β_jac, β); max_W2_β = maximum(W2_β);
    W2_T = W2_sample_distance(T_jac, T); max_W2_T = maximum(vcat(vec.(W2_T)...));

    @test max_W2_β < 1e-3;
    @test max_W2_T < 1e-3;
end;