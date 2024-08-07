using ASLA: PGgibbs_GLMMs, JacobiPreconditioner, ICholPreconditioner, W2_sample_distance

using IterativeSolvers
using Random, Distributions
using CSV, DataFrames, DataStructures
using MixedModels, StatsModels, StatsBase
using Tables
using JLD2, FileIO
using PrettyTables

path_to_folder = "paper\\ichol_preconditioner\\";
path_to_data = "paper\\real_data_example\\data\\";

df = DataFrame(CSV.File(path_to_data*"dat_cps_2004.csv", delim=","));
column_indeces = vcat(2:6, 17, 19:22);
df = df[:, column_indeces]

# Dataframes
function ER_factors(N::Integer, I::Vector{R}; remove_unobserved_levels::Bool=false, Tv=Float64, Ti=Int64, seed=nothing, debug::Bool=false) where{R<:Integer}
    K_dim_array_size = prod(I)

    samples_from_mult_array = sample(1:K_dim_array_size, N, replace=false)

    cartesian = CartesianIndices(Tuple(I))
    return transpose(hcat([collect(Tuple(cartesian[sample])) for sample in samples_from_mult_array]...))
end


function get_dataframes(df::DataFrame, N::Int; graph_type="indep_factors")
    # Real 
    df_real = df[sample(1:size(df)[1], N, replace=false, ordered=true), :];

    # Simulated
    if graph_type == "ER" # only works if N<prod(I)
        # ER random graph
        I = [5, 4, 51, 4, 5]; 
        @assert N<prod(I) "N cannot prod(I) in ER-type graph"
        aux = ER_factors(N, I);  # matrix with factors 
        y = zeros(Int64, N); z_inc = standardize(ZScoreTransform, convert.(Float64, aux[:, 1])); z_incstt = rand(Normal(0., 1.), N); z_trnprv = standardize(ZScoreTransform, rand(Gamma(12.,4.), N));
        df_sim = DataFrame(
            :vote => y,
            :inc => aux[:, 1], :eth => aux[:, 2], :stt => aux[:, 3], :age => aux[:, 4], :reg => aux[:, 5],
            :z_inc => z_inc, :z_incstt => z_incstt, :z_trnprv => z_trnprv
        );
    else
        inc = sample(1:5, N); eth = sample(1:4, N); stt = sample(1:51, N); age = sample(1:4, N); reg = sample(1:5, N);
        z_inc = standardize(ZScoreTransform, convert.(Float64, inc)); z_incstt = rand(Normal(0., 1.), N); z_trnprv = standardize(ZScoreTransform, rand(Gamma(12.,4.), N));
        vote = zeros(Int64, N);

        df_sim = DataFrame(
            :vote => vote,
            :inc => inc, :eth => eth, :stt => stt, :age => age, :reg => reg,
            :z_inc => z_inc, :z_incstt => z_incstt, :z_trnprv => z_trnprv
        );
    end
    

    f = @formula(vote ~ 1 + z_inc + (1 | inc) + (1 | eth) + (1 | stt) + (1 | age));

    tbl = Tables.columntable(df_sim);
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}());
    _, Xs = MixedModels.modelcols(form, tbl);

    V = hcat(sparse.(Xs[2:end])..., Xs[1]);
    p = sum([size(x)[2] for x in Xs]); G_0 = size(Xs[1])[2]; α = rand(Normal(0.0, 0.5), p - G_0); β = [0.6, -0.7]; θ = vcat(α, β);

    logit(t) = exp(t) / (1 + exp(t))
    vote = rand.(Bernoulli.(logit.(V * θ)));
    df_sim[:, :vote] = vote;
    return df_sim, df_real
end

## AUXILIARY FUNCTIONS
function cg_results(df::DataFrame, f::FormulaTerm, n_iters::Int; burn_in::Int=1, seed=16, accuracy_cg=1e-8)
    @assert n_iters > burn_in

    function _cg!(x, Q, b, iters, Pl)
        # global iters
        _, ch = cg!(x, Q, b, log=true, reltol=accuracy_cg, Pl=Pl)
        push!(iters, ch.iters)
        return nothing
    end

    converged_values = PGgibbs_GLMMs(df, f, burn_in, seed=10, converged_values=true)

    iters_jacobi = Int[];    iters_ichol = Int[];

    _, _ = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! = (x, Q, b) -> _cg!(x,Q, b, iters_jacobi, JacobiPreconditioner(Q)))
    _, _ = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! = (x, Q, b) -> _cg!(x,Q, b, iters_ichol, ICholPreconditioner(Q)))

    return Dict(
        :iters_jacobi => iters_jacobi,
        :iters_ichol => iters_ichol,
        :size => length(converged_values[:θ])
    )
end;


#### RESULTS ####
formula_list = [
    # Random intercepts
    @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc)),
    
    # Adding nested effect (stt into reg)
    @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc) + (1|reg)),
    
    # Random slopes 
    @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | inc) + (1 + z_inc | eth) + (1 + z_inc | stt) + (1 + z_inc | age)),
    
    # Random slopes (with nested factors)
    # @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | inc) + (1 + z_inc | eth) + (1 + z_inc | stt) + (1 + z_inc | age) + (1|reg)),
    
    # 2 way interactions
    @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc)+ (1 | eth&inc) + (1 | eth&age) + (1 | inc&age) + (1 | stt&eth) + (1 | stt&inc) + (1 | stt&age)),
    
    # 3 way interactions
    @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | age) + (1 | eth) + (1 | stt) + (1 | inc) + (1 | eth&inc) + (1 | eth&age) + (1 | inc&age) + (1 | stt&eth) + (1 | stt&inc) + (1 | stt&age) + (1 | eth&inc&age) + (1 | stt&eth&inc) + (1 | stt&eth&age) + (1 | stt&inc&age)),
    
    # Everything
    @formula(vote ~  z_inc + z_incstt + z_trnprv + (1 | inc) + (1 + z_inc | eth) + (1 + z_inc | stt) + (1 + z_inc | age) + (1|reg) + (1 | eth&inc) + (1 | eth&age) + (1 | inc&age) + (1 | stt&eth) + (1 | stt&inc) + (1 | stt&age) + (1 | eth&inc&age) + (1 | stt&eth&inc) + (1 | stt&eth&age) + (1 | stt&inc&age))
];


if false
    for N in [70000]
        df_sim, df_real = get_dataframes(df, N)
        results_sim  = DataFrame(iters_jacobi = Float64[], iters_ichol = Float64[], size = Int[]);
        results_real = DataFrame(iters_jacobi = Float64[], iters_ichol = Float64[], size = Int[]);
        outputs_sim = []; outputs_real = [];

        n_iters = 1000; burn_in = 500;
        for f in formula_list
            println(f)
            
            out = cg_results(df_sim, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8)
            push!(results_sim, (mean(out[:iters_jacobi]), mean(out[:iters_ichol]), out[:size]))
            push!(outputs_sim, out)

            out = cg_results(df_real, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8)
            push!(results_real, (mean(out[:iters_jacobi]), mean(out[:iters_ichol]), out[:size]))
            push!(outputs_real, out)

        end
        println(results_real)

        FileIO.save(path_to_folder*"results_GG.jld2", "results_sim", results_sim, "results_real", results_real, "outputs_sim", outputs_sim, "outputs_real", outputs_real)
    end
end


results_sim, outputs_sim, results_real, outputs_real = load(path_to_folder*"results_GG.jld2", "results_sim", "outputs_sim", "results_real", "outputs_real");

jacobi_sim = round.(Int64, results_sim.iters_jacobi); ichol_sim = round.(Int64, results_sim.iters_ichol); size_sim = results_sim.size
jacobi_real = round.(Int64, results_real.iters_jacobi); ichol_real = round.(Int64, results_real.iters_ichol); size_real = results_real.size

cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]

pretty_df = DataFrame(
    Case = cases,
    Real_jac = jacobi_real,
    Real_ichol = ichol_real,
    Real_size = size_real,
    Sim_jac = jacobi_sim,
    Sim_ichol = ichol_sim,
    Sim_size = size_sim,
)

CSV.write(path_to_folder*"summary_GG.csv", pretty_df)