using ASLA: PGgibbs_GLMMs, JacobiPreconditioner, W2_sample_distance

using LinearAlgebra, SparseArrays
using IterativeSolvers
using Random, Distributions
using CSV, DataFrames, DataStructures
using MixedModels, StatsModels, StatsBase
using Tables
using JLD2, FileIO
using PrettyTables

path_to_folder = "paper\\different_stopping_criteria\\"
path_to_data = "paper\\real_data_example\\data\\"

df = DataFrame(CSV.File(path_to_data*"insteval.csv", delim=","))[:, 2:end]; df.y .-=1;

# Dataframes
function get_dataframes(df::DataFrame, N::Int)
    # Real 
    df_real = df[1:N, :];

    # Simulated
    F1 = sample(1:2972, N); F2 = sample(1:1128, N); F3 = sample(1:4, N); F4 = sample(1:6, N); F5 = sample(1:14, N); 
    y = zeros(Int64, N);

    df_sim = DataFrame(
        :y => y,
        :s => F1, :d => F2, :studage => F3, :lectage => F4, :dept => F5);
    f = @formula(y ~ 1 + (1 | s) + (1 | d) + (1 | lectage) + (1 | dept));

    tbl = Tables.columntable(df_sim);
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}());
    _, Xs = MixedModels.modelcols(form, tbl);

    V = hcat(sparse.(Xs[2:end])..., Xs[1]);
    p = sum([size(x)[2] for x in Xs]); G_0 = size(Xs[1])[2]; α = rand(Normal(0.0, 0.5), p - G_0); β = [0.6]; θ = vcat(α, β);

    logit(t) = exp(t) / (1 + exp(t))
    y = rand.(Binomial.(4, logit.(V * θ)));
    df_sim[:, :y] = y;
    return df_sim, df_real
end

## AUXILIARY FUNCTIONS
function cg_results(df::DataFrame, f::FormulaTerm, n_iters::Int; burn_in::Int=1, return_W2::Bool=false, seed=121, accuracy_cg=1e-5)
    @assert n_iters > burn_in

    function _cg!(x, Q, b; tol=accuracy_cg)
        iterable = IterativeSolvers.cg_iterator!(deepcopy(x), Q, b, JacobiPreconditioner(Q), reltol=0.0, maxiter = 200)
        x = Q\b;
        global iters
        for (iteration, _) = enumerate(iterable)
            if maximum(abs.(x- iterable.x)) < tol
                push!(iters, iteration)
                break
            end
        end
        return nothing
    end

    converged_values = PGgibbs_GLMMs(df, f, burn_in, seed=10, converged_values=true)

    β_cg, T_cg = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! =_cg!)

    if return_W2
        β_exact, T_exact = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed)
        return Dict(
            :accuracy => accuracy_cg,
            :iters => iters,
            :size => length(converged_values[:θ]),
            :W2_β => W2_sample_distance(β_cg, β_exact),
            :W2_T => W2_sample_distance(T_cg, T_exact)
        )
    end

    return Dict(
        :accuracy => accuracy_cg,
        :iters => iters,
        :size => length(converged_values[:θ])
    )
    return nothing
end;


function cg_results2(df::DataFrame, f::FormulaTerm, n_iters::Int; burn_in::Int=1, return_W2::Bool=false, seed=121, accuracy_cg=[1e-5, 1e-9, 1e-9])
    @assert n_iters > burn_in

    function _cg!(x, Q, b; tol=accuracy_cg)
        iterable = IterativeSolvers.cg_iterator!(deepcopy(x), Q, b, JacobiPreconditioner(Q), reltol=0.0, maxiter = 200)
        x = Q\b;
        global iters1, iters2, iters3
        conv1 = 0; conv2 = 0; conv3 = 0;
        for (iteration, _) = enumerate(iterable)
            if conv1 ==0 &&  maximum(abs.(x- iterable.x)) < tol[1]
                push!(iters1, iteration)
                conv1 = 1
            end
            if conv2 ==0 &&  norm(x- iterable.x)/norm(x) < tol[2]
                push!(iters2, iteration)
                conv2 = 1
            end
            if conv3 ==0 &&  iterable.residual/norm(b) < tol[3]
                push!(iters3, iteration)
                conv3 = 1
            end

            
            conv1*conv2*conv3==1 && break
        end
        return nothing
    end

    converged_values = PGgibbs_GLMMs(df, f, burn_in, seed=10, converged_values=true)

    β_cg, T_cg = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! =_cg!)

    if return_W2
        β_exact, T_exact = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed)
        return Dict(
            :accuracy => accuracy_cg,
            :iters1 => iters1,
            :iters2 => iters2,
            :iters3 => iters3,
            :size => length(converged_values[:θ]),
            :W2_β => W2_sample_distance(β_cg, β_exact),
            :W2_T => W2_sample_distance(T_cg, T_exact)
        )
    end

    return Dict(
        :accuracy => accuracy_cg,
        :iters1 => iters1,
        :iters2 => iters2,
        :iters3 => iters3,
        :size => length(converged_values[:θ])
    )
end;

function _get_PG_Q(df::DataFrame, f::FormulaTerm, T, Ω)
    tbl = Tables.columntable(df)
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}())
    _, Xs = MixedModels.modelcols(form, tbl)

    V = hcat(sparse.(Xs[2:end])..., Xs[1]); V_T = transpose(V)

    G_0 = size(Xs[1])[2]; N = length(y); D = _get_D(form); G = _get_G(Xs, D); K = length(G); p = sum([size(x)[2] for x in Xs])

    prior_prec = blockdiag(vcat([repeat([sparse(T_k)], G_k) for (T_k, G_k) in zip(T, G)]...)..., spzeros(G_0, G_0))
    return prior_prec + V_T*spdiagm(Ω)*V
end

function _get_PG_b(df, f, T, Ω)
    tbl = Tables.columntable(df)
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}())
    y, Xs = MixedModels.modelcols(form, tbl)

    n = length(unique(y))-1;  y_centered = y .- 0.5*n
    G_0 = size(Xs[1])[2]; N = length(y); D = _get_D(form); G = _get_G(Xs, D); K = length(G); p = sum([size(x)[2] for x in Xs])


    V = hcat(sparse.(Xs[2:end])..., Xs[1]); V_T = transpose(V)
    T_sqrt = [cholesky(Symmetric(T_k), NoPivot()).L for T_k in T]
    prior_prec_sqrt = blockdiag(vcat([repeat([sparse(L_k)], G_k) for (L_k, G_k) in zip(T_sqrt, G)]...)..., spzeros(G_0, G_0))
    
    return V_T*y_centered + V_T * (sqrt.(Ω) .* rand(Normal(0, 1), N)) + prior_prec_sqrt*rand(Normal(0,1), p) 
end

function _sample_Q_a_posteriori(df::DataFrame, f::FormulaTerm, n_iters::Int)
    converged_values = PGgibbs_GLMMs(df, f, n_iters, seed=10, converged_values=true, (system_solver!)=(x, Q, b) -> cg!(x, Q, b, Pl=Diagonal(diag(Q))))

    T = converged_values[:T]
    Ω = converged_values[:Ω]
    return _get_PG_Q(df, f, T, Ω)
end


#### RESULTS ####
formula_list = [
    # Random intercepts
    @formula(y ~ 1 + (1 | s) + (1 | d)),
    
    # Adding nested effect (dept into d)
    @formula(y ~ 1 + (1 | s) + (1| d) + (1|dept)),

    # Adding random slopes (no nested)
    @formula(y ~ 1 + (1 | s) + (1+studage| d)),

    # Adding random slopes (with nested)
    @formula(y ~ 1 + (1 | s) + (1+studage| d) + (1|dept)),

    # 2 way interactions
    @formula(y ~ 1 + (1 | s) + (1 | d) + (1 | lectage) + (1 | s&d) + (1 | s&lectage)),

    # 3 way interactions
    @formula(y ~ 1 + (1 | s) + (1 | d) + (1 | lectage) + (1|studage) + (1 | s&studage) + (1 | s&lectage) + (1 | s&lectage&studage))
];


iters1 = []; iters2 = []; iters3 = [];
if true
    for N in [5000, 70000]
        df_sim, df_real = get_dataframes(df, N)
        # results_sim = DataFrame(iters = Float64[], size = Int[], max_W2_β = Float64[], max_W2_T = Float64[], avg_W2_β = Float64[], avg_W2_T = Float64[]);
        # results_real = DataFrame(iters = Float64[], size = Int[], max_W2_β = Float64[], max_W2_T = Float64[], avg_W2_β = Float64[], avg_W2_T = Float64[]);
        results_sim =   DataFrame(iters1 = Float64[], iters2 = Float64[], iters3 = Float64[], size = Int[]);
        results_real =  DataFrame(iters1 = Float64[], iters2 = Float64[], iters3 = Float64[], size = Int[]);
        outputs_sim = []; outputs_real = [];

        n_iters = 300; burn_in = 100;
        for f in formula_list
            println(f)
            
            global iters1, iters2, iters3
            iters1 = []; iters2 = []; iters3 = [];
            out = cg_results2(df_sim, f, n_iters; burn_in=burn_in, return_W2=false, accuracy_cg=[1e-5, 3*1e-6, 1e-8])
            # W2_T = vcat(vec.(out[:W2_T])...); W2_β = out[:W2_β]; push!(results_sim, (mean(out[:iters]), out[:size], maximum(W2_β), maximum(W2_T), mean(W2_β), mean(W2_T)))
            push!(results_sim, (mean(iters1), mean(iters2), mean(iters3), out[:size]))
            push!(outputs_sim, out)

            iters1 = []; iters2 = []; iters3 = [];
            out = cg_results2(df_real, f, n_iters; burn_in=burn_in, return_W2=false, accuracy_cg=[1e-5, 3*1e-6, 1e-8])
            # W2_T = vcat(vec.(out[:W2_T])...); W2_β = out[:W2_β]; push!(results_real, (mean(out[:iters]), out[:size], maximum(W2_β), maximum(W2_T), mean(W2_β), mean(W2_T)))
            push!(results_real, (mean(iters1), mean(iters2), mean(iters3), out[:size]))
            push!(outputs_real, out)

        end
        println(results_sim)
        println(results_real)

        N <= 10000 && FileIO.save(path_to_folder*"results_different_stopping_criterion.jld2", "results_sim", results_sim, "outputs_sim", outputs_sim, "results_real", results_real, "outputs_real", outputs_real)
        N >= 50000 && FileIO.save(path_to_folder*"results_largeN_different_stopping_criterion.jld2", "results_sim", results_sim, "outputs_sim", outputs_sim, "results_real", results_real, "outputs_real", outputs_real)
    end
end


results_sim, _, results_real, _ = load(path_to_folder*"results_different_stopping_criterion.jld2", "results_sim", "outputs_sim", "results_real", "outputs_real");
results_sim_large, results_real_large = load(path_to_folder*"results_largeN_different_stopping_criterion.jld2", "results_sim", "results_real");