using ASLA: PGgibbs_GLMMs, JacobiPreconditioner

using IterativeSolvers
using Random, Distributions
using CSV, DataFrames, DataStructures
using MixedModels, StatsModels, StatsBase
using Tables
using JLD2, FileIO
using PrettyTables
using LinearAlgebra, SparseArrays

path_to_folder = joinpath("paper", "real_data_example", "python")
path_to_data = joinpath("paper", "real_data_example", "data")

dataset = "GG"

if dataset == "GG"
    df = DataFrame(CSV.File(joinpath(path_to_data, "dat_cps_2004.csv"), delim=","));
    column_indeces = vcat(2:6, 17, 19:22);
    df = df[:, column_indeces]

    formula_list = [
        # Random intercepts
        @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc)),
        
        # Adding nested effect (stt into reg)
        @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc) + (1|reg)),
        
        # Random slopes 
        @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | inc) + (1 + z_inc | eth) + (1 + z_inc | stt) + (1 + z_inc | age)),
        
        # 2 way interactions
        @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc)+ (1 | eth&inc) + (1 | eth&age) + (1 | inc&age) + (1 | stt&eth) + (1 | stt&inc) + (1 | stt&age)),
        
        # 3 way interactions
        @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc) + (1 | eth&inc) + (1 | eth&age) + (1 | inc&age) + (1 | stt&eth) + (1 | stt&inc) + (1 | stt&age) + (1 | eth&inc&age) + (1 | stt&eth&inc) + (1 | stt&eth&age) + (1 | stt&inc&age)),
        
        # Everything
        @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | inc) + (1 + z_inc | eth) + (1 + z_inc | stt) + (1 + z_inc | age) + (1|reg) + (1 | eth&inc) + (1 | eth&age) + (1 | inc&age) + (1 | stt&eth) + (1 | stt&inc) + (1 | stt&age) + (1 | eth&inc&age) + (1 | stt&eth&inc) + (1 | stt&eth&age) + (1 | stt&inc&age))
    ];

elseif dataset == "IE"

    df = DataFrame(CSV.File(joinpath(path_to_data,"insteval.csv"), delim=","))[:, 2:end]; df.y .-=1;
    formula_list = [
        # Random intercepts
        @formula(y ~ 1 + (1 | s) + (1 | d)),
        
        # Adding nested effect (dept into d)
        @formula(y ~ 1 + (1 | s) + (1| d) + (1|dept)),

        # Adding random slopes (no nested)
        @formula(y ~ 1 + (1 | s) + (1+service| d)),

        # 2 way interactions
        @formula(y ~ 1 + (1 | s) + (1 | d) + (1 | lectage) + (1 | s&d) + (1 | s&lectage) + (1 | d&lectage))
    ];
end
for (index, f) in enumerate(formula_list)

    initial_values = PGgibbs_GLMMs(df, f, 30, seed=10, converged_values=true, system_solver! = (x, Q, b) -> cg!(x,Q,b))

    # building Q
    tbl = Tables.columntable(df)
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}())
    y, Xs = MixedModels.modelcols(form, tbl)
    n = length(unique(y))-1 # where y ~ Binom(n, x' β)

    V = hcat(sparse.(Xs[2:end])..., Xs[1]) # fixed effects in last position
    V_T = transpose(V)
    y_centered = y .- 0.5*n

    _get_D(form::FormulaTerm) = [length(rand_eff.lhs.terms) for rand_eff in form.rhs[2:end]]

    _get_G(Xs::Tuple, D::Vector{Int64}) = [size(x)[2] for x in Xs[2:end]] .÷ D

    G_0 = size(Xs[1])[2]; N = length(y); D = _get_D(form); G = _get_G(Xs, D); K = length(G); p = sum([size(x)[2] for x in Xs])

    θ = deepcopy(initial_values[:θ])
    T = deepcopy(initial_values[:T])
    Ω = deepcopy(initial_values[:Ω])

    prior_prec = blockdiag(vcat([repeat([sparse(T_k)], G_k) for (T_k, G_k) in zip(T, G)]...)..., spzeros(G_0, G_0))
    T_sqrt = [cholesky(Symmetric(T_k), NoPivot()).L for T_k in T]
    prior_prec_sqrt = blockdiag(vcat([repeat([sparse(L_k)], G_k) for (L_k, G_k) in zip(T_sqrt, G)]...)..., spzeros(G_0, G_0))

    Q = prior_prec + V_T*spdiagm(Ω)*V


    i, j, v = findnz(Q)
    df_aux = DataFrame([:I => i, :J => j, :V => v])
    θ_col = Vector{Union{Float64, Missing}}(missing, nrow(df_aux))
    θ_len = min(length(θ), nrow(df_aux))
    θ_col[1:θ_len] .= θ[1:θ_len]
    df_aux.:th = θ_col
    CSV.write(joinpath(path_to_folder, "spmatrix_$(dataset)$(index).csv"), df_aux)
    CSV.write(joinpath(path_to_folder, "spmatrix_$(dataset)$(index).csv"), df_aux)
end


# Running times CG and Chol in Julia
results_real = load(joinpath("paper", "real_data_example", "G&G", "results_largeN.jld2"), "results_real")

cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]

time_df = DataFrame(
    Case = cases,
    Cholesky = results_real.time_chol,
    CG = results_real.time_cg,
)