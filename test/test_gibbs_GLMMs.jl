using CSV
using Profile
using MixedModels, StatsModels, StatsBase
using DataFrames


### test on simulated data
@testset "Simulated data Gibbs" begin
    N = 5000
    X1 = rand(Normal(0., 1.), N)
    X2 = standardize(ZScoreTransform, rand(Gamma(12.,4.), N))

    F1 = sample(1:10, N); F2 = sample(1:5, N)

    y = zeros(Int64, N) 

    df = DataFrame(:y  => y, :X1 => X1, :X2 => X2, :F1 => F1, :F2 => F2)

    f = @formula(y ~ 1 + X1 + X2 + (1|F1) + (1+X1|F2) + (1|F1&F2))

    tbl = Tables.columntable(df); form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}()); _, Xs = MixedModels.modelcols(form, tbl)

    V = hcat(sparse.(Xs[2:end])..., Xs[1]); p = sum([size(x)[2] for x in Xs]); G_0 = size(Xs[1])[2]; α = rand(Normal(0., 0.5), p-G_0); β = [3., 2.3, -1.4]; θ = vcat(α, β)

    logit(t) = exp(t)/(1+exp(t)); y = rand.(Bernoulli.(logit.(V*θ))); df[:, :y] = y

    n_iters = 500; burn_in = 200;
    β_hist, T_hist = PGgibbs_GLMMs(df, f, n_iters, burn_in=burn_in);

    @test length(β_hist)==length(T_hist)
    @test length(β_hist)== n_iters - burn_in + 1
end;


#### Test with MixedModels datasets ####
@testset "Real data Gibbs" begin
    verbagg = MixedModels.dataset(:verbagg)
    f = @formula(r2 ~ 1 + anger + gender + btype + situ + (1|subj) + (1|item) + (1|subj&item));

    n_iters = 500; burn_in = 200;
    β_hist, T_hist = PGgibbs_GLMMs(DataFrame(verbagg), f, n_iters, burn_in=burn_in);

    @test length(β_hist)==length(T_hist)
    @test length(β_hist[1])==6 # since btype is categorical with 3 categories.
end;