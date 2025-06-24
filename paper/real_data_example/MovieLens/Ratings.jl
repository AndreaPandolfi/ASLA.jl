using ASLA: PGgibbs_GLMMs, JacobiPreconditioner, W2_sample_distance

using IterativeSolvers
using Random, Distributions
using CSV, DataFrames, DataStructures
using MixedModels, StatsModels, StatsBase
using Tables
using JLD2, FileIO
using PrettyTables

path_to_folder = joinpath("paper", "real_data_example", "MovieLens")
path_to_data = # download ml-25m from https://grouplens.org/datasets/movielens/ and specify the path to the ratings.csv file


df = DataFrame(CSV.read(joinpath(path_to_data, "ratings.csv"), DataFrame))[:, 1:end-1];
df.rating = round.(Int, df.rating .* 2 .-1); # transform rating to Int b/w 0 and 9


# Dataframes
get_df_sel(df::DataFrame, N::Int) = sample(1:size(df)[1], N, replace=false, ordered=true);


function ER_factors(N::Integer, I::Vector{R}; remove_unobserved_levels::Bool=false, Tv=Float64, Ti=Int64, seed=nothing, debug::Bool=false) where{R<:Integer}
    K_dim_array_size = prod(I)

    samples_from_mult_array = sample(1:K_dim_array_size, N, replace=false)

    cartesian = CartesianIndices(Tuple(I))
    return transpose(hcat([collect(Tuple(cartesian[sample])) for sample in samples_from_mult_array]...))
end

function get_dataframes(df::DataFrame, N::Int; graph_type="ER")
    # Real 
    # df_real = df[sample(1:size(df)[1], N, replace=false, ordered=true), :];
    df_real = df[get_df_sel(df, N), :];
    
    N = size(df_real)[1]; 
    I = [length(unique(df_real.userId)), length(unique(df_real.movieId))]; 
    if graph_type == "ER"
        # ER random graph
        aux = ER_factors(N, I); rating = zeros(Int64, N);
        
        # Enforcing presence of each factor to have fixed p
        aux[1:I[1], 1] = sample(1:I[1], I[1], replace=false)
        aux[1:I[2], 2] = sample(1:I[2], I[2], replace=false)


        df_sim = DataFrame(
            :rating => rating,
            :userId => aux[:, 1], :movieId => aux[:, 2]
        );
    else
        # Sample each factor independently
        F1 = vcat(sample(1:I[1], N-I[1]), sample(1:I[1], I[1], replace=false)); F2 = vcat(sample(1:I[2], N-I[2]), sample(1:I[2], I[2], replace=false));
        rating = zeros(Int64, N);

        df_sim = DataFrame(
            :rating => rating,
            :userId => F1, :movieId => F2
        );
    end
    f = @formula(rating ~ 1 + (1 | userId) + (1 | movieId))

    tbl = Tables.columntable(df_sim);
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}());
    _, Xs = MixedModels.modelcols(form, tbl);

    V = hcat(sparse.(Xs[2:end])..., Xs[1]);
    p = sum([size(x)[2] for x in Xs]); G_0 = size(Xs[1])[2]; α = rand(Normal(0.0, 0.5), p - G_0); β = [0.6]; θ = vcat(α, β);

    logit(t) = exp(t) / (1 + exp(t))
    rating = rand.(Binomial.(4, logit.(V * θ)));
    df_sim[:, :rating] = rating;
    return df_sim, df_real
end

## AUXILIARY FUNCTIONS
function cg_results(df::DataFrame, f::FormulaTerm, n_iters::Int; burn_in::Int=1, seed=121, accuracy_cg=1e-8, time_chol=false)
    @assert n_iters > burn_in

    function _cg!(x, Q, b)
        t = @elapsed _, ch = cg!(x, Q, b, log=true, reltol=accuracy_cg, Pl=JacobiPreconditioner(Q))
        cost_cg = cost_per_iter_CG(Q) * ch.iters
        return Dict(
            :iters => ch.iters,
            :cost => cost_cg,
            :elapsed => t
        )
    end

    converged_values = PGgibbs_GLMMs(df, f, burn_in, seed=10, converged_values=true, system_solver! = (x, Q, b) -> cg!(x,Q,b))

    β_cg, T_cg, info_hist = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! =_cg!, runtime_info=true)

    iters = [info[:iters] for info in info_hist]
    cost_cg = [info[:cost] for info in info_hist]
    elapsed_cg = [info[:elapsed] for info in info_hist]

    if time_chol
        function _chol!(x, Q, b)
            t = @elapsed F = cholesky(Symmetric(Q)); x .= F \ b
            cost_chol = cost_cholesky(sparse(F.L)) # cost_cholesky(F.L, F.piv)
            nnz_Q = length(Q.nzval)
            nnz_L = length(sparse(F.L).nzval)
            return Dict(
                :cost => cost_chol,
                :elapsed => t, 
                :nnz_Q => nnz_Q,
                :nnz_L => nnz_L
            )
        end
        β_exact, T_exact, info_hist = PGgibbs_GLMMs(df, f, 5, initial_values=converged_values, seed=seed, system_solver! =_chol!, runtime_info=true)
        cost_chol = [info[:cost] for info in info_hist]
        elapsed_chol = [info[:elapsed] for info in info_hist]
        nnz_Q = [info[:nnz_Q] for info in info_hist]
        nnz_L = [info[:nnz_L] for info in info_hist]
        return Dict(
            :accuracy => accuracy_cg,
            :iters => iters,
            :elapsed_cg => elapsed_cg,
            :elapsed_chol => elapsed_chol,
            :cost_cg => cost_cg,
            :cost_chol => cost_chol,
            :size => length(converged_values[:θ]),
            :nnz_Q => nnz_Q,
            :nnz_L => nnz_L
        )
    end

    return Dict(
        :accuracy => accuracy_cg,
        :iters => iters,
        :elapsed_cg => elapsed_cg,
        :cost_cg => cost_cg,
        :size => length(converged_values[:θ])
    )
end;



#### RESULTS ####
formula_list = [
    # Random intercepts
    @formula(rating ~ 1 + (1 | userId) + (1 | movieId)),
];

if false
    for N in [25000, 25000000]
        results_sim = DataFrame(iters = Float64[], size = Int[]);
        results_real = DataFrame(iters = Float64[], size = Int[]);
        outputs_sim = []; 
        outputs_real = [];
        df_sim, df_real = get_dataframes(df, N)

        n_iters = 300; burn_in = 100;
        for f in formula_list
            println(f)

            if N <= 300000
                out = cg_results(df_real, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8, time_chol=true)
                push!(results_real, (mean(out[:iters]), out[:size]), mean(out[:elapsed_cg]), mean(out[:elapsed_chol]), mean(out[:cost_cg]), mean(out[:cost_chol]), mean(out[:nnz_Q]), mean(out[:nnz_L]))
            else
                out = cg_results(df_real, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8)
                push!(results_real, (mean(out[:iters]), out[:size]))
            end
            push!(outputs_real, out)

            out = cg_results(df_sim, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8)
            push!(results_sim, (mean(out[:iters]), out[:size]))
            push!(outputs_sim, out)
        end
        println(results_real)

        N <= 300000 && FileIO.save(joinpath(path_to_folder, "results.jld2"), "results_sim", results_sim, "outputs_sim", outputs_sim, "results_real", results_real, "outputs_real", outputs_real)
        N >= 300000 && FileIO.save(joinpath(path_to_folder, "results_largeN.jld2"), "results_sim", results_sim, "results_real", results_real, "outputs_sim", outputs_sim, "outputs_real", outputs_real)
    end
end


results_sim, _, results_real, _ = load(joinpath(path_to_folder, "results.jld2"), "results_sim", "outputs_sim", "results_real", "outputs_real");
results_sim_large, results_real_large = load(joinpath(path_to_folder, "results_largeN.jld2"), "results_sim", "results_real");


iters_sim = ones(2*length(results_sim.iters));      iters_sim[1:2:end] .= results_sim.iters;    iters_sim[2:2:end] .= results_sim_large.iters;      iters_sim = round.(Int64, iters_sim);
size_sim = ones(2*length(results_sim.iters));       size_sim[1:2:end] .= results_sim.size;      size_sim[2:2:end] .= results_sim_large.size;        size_sim = round.(Int64, size_sim);
iters_real = ones(2*length(results_real.iters));    iters_real[1:2:end] .= results_real.iters;  iters_real[2:2:end] .= results_real_large.iters;    iters_real = round.(Int64, iters_real);
size_real = ones(2*length(results_real.size));      size_real[1:2:end] .= results_real.size;    size_real[2:2:end] .= results_real_large.size;      size_real = round.(Int64, size_real);
cases = ["Random intercepts"]


pretty_df = DataFrame(
    Case = vcat([[case, ""] for case in cases]...),
    Real = ["$(iters) ($(size))" for (iters, size) in zip(iters_real, size_real)],
    Simulated = ["$(iters) ($(size))" for (iters, size) in zip(iters_sim, size_sim)]
)

CSV.write(joinpath(path_to_folder, "summary_Ratings.csv"), pretty_df)