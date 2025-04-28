using ASLA: PGgibbs_GLMMs, JacobiPreconditioner, W2_sample_distance, cost_per_iter_CG, cost_cholesky

using IterativeSolvers
using Random, Distributions
using CSV, DataFrames
using MixedModels, StatsModels, StatsBase
using Tables
using JLD2, FileIO
using PrettyTables
using LinearAlgebra


path_to_folder = joinpath("paper", "real_data_example", "G&G")
path_to_data = joinpath("paper", "real_data_example", "data")


df = DataFrame(CSV.File(joinpath(path_to_data, "dat_cps_2004.csv"), delim=","));
column_indeces = vcat(2:6, 17, 19:22);
df = df[:, column_indeces]

# Dataframes
function get_dataframes(df::DataFrame, N::Int)
    # Real 
    df_real = df[sample(1:size(df)[1], N, replace=false, ordered=true), :];

    # Simulated
    inc = sample(1:5, N); eth = sample(1:4, N); stt = sample(1:51, N); age = sample(1:4, N); reg = sample(1:5, N);
    z_inc = standardize(ZScoreTransform, convert.(Float64, inc)); z_incstt = rand(Normal(0., 1.), N); z_trnprv = standardize(ZScoreTransform, rand(Gamma(12.,4.), N));
    vote = zeros(Int64, N);

    df_sim = DataFrame(
        :vote => vote,
        :inc => inc, :eth => eth, :stt => stt, :age => age, :reg => reg,
        :z_inc => z_inc, :z_incstt => z_incstt, :z_trnprv => z_trnprv
    );

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
function results(df::DataFrame, f::FormulaTerm, n_iters::Int; burn_in::Int=1, return_W2::Bool=false, seed=121, accuracy_cg=1e-8)
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

    converged_values = PGgibbs_GLMMs(df, f, burn_in, seed=10, converged_values=true)

    β_cg, T_cg, info_hist = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! =_cg!, runtime_info=true)

    iters = [info[:iters] for info in info_hist]
    cost_cg = [info[:cost] for info in info_hist]
    elapsed_cg = [info[:elapsed] for info in info_hist]

    if return_W2
        function _chol!(x, Q, b)
            t = @elapsed F = cholesky(Symmetric(Q)); x .= F \ b
            cost_chol = cost_cholesky(sparse(F.L)) # cost_cholesky(F.L, F.piv)
            return Dict(
                :cost => cost_chol,
                :elapsed => t
            )
        end
        β_exact, T_exact, info_hist = PGgibbs_GLMMs(df, f, n_iters - burn_in, initial_values=converged_values, seed=seed, system_solver! =_chol!, runtime_info=true)
        cost_chol = [info[:cost] for info in info_hist]
        elapsed_chol = [info[:elapsed] for info in info_hist]
        return Dict(
            :accuracy => accuracy_cg,
            :iters => iters,
            :elapsed_cg => elapsed_cg,
            :elapsed_chol => elapsed_chol,
            :cost_cg => cost_cg,
            :cost_chol => cost_chol,
            :size => length(converged_values[:θ]),
            :W2_β => W2_sample_distance(β_cg, β_exact),
            :W2_T => W2_sample_distance(T_cg, T_exact)
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
    for N in [7000, 70000]
        df_sim, df_real = get_dataframes(df, N)
        results_sim = DataFrame(iters = Float64[], size = Int[], time_cg = Float64[], time_chol = Float64[], cost_cg = Float64[], cost_chol = Float64[], max_W2_β = Float64[], max_W2_T = Float64[], avg_W2_β = Float64[], avg_W2_T = Float64[]);
        results_real = DataFrame(iters = Float64[], size = Int[], time_cg = Float64[], time_chol = Float64[], cost_cg = Float64[], cost_chol = Float64[], max_W2_β = Float64[], max_W2_T = Float64[], avg_W2_β = Float64[], avg_W2_T = Float64[]);
        outputs_sim = []; outputs_real = [];

        n_iters = 300; burn_in = n_iters÷3;
        for f in formula_list
            println(f)
            
            out = results(df_sim, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8, return_W2=true)
            W2_T = vcat(vec.(out[:W2_T])...); W2_β = out[:W2_β];
            push!(results_sim, (mean(out[:iters]), out[:size], mean(out[:elapsed_cg]), mean(out[:elapsed_chol]), mean(out[:cost_cg]), mean(out[:cost_chol]),  maximum(W2_β), maximum(W2_T), mean(W2_β), mean(W2_T)))
            push!(outputs_sim, out)

            out = results(df_real, f, n_iters; burn_in=burn_in, accuracy_cg=1e-8, return_W2=true)
            W2_T = vcat(vec.(out[:W2_T])...); W2_β = out[:W2_β];
            push!(results_real, (mean(out[:iters]), out[:size], mean(out[:elapsed_cg]), mean(out[:elapsed_chol]), mean(out[:cost_cg]), mean(out[:cost_chol]),  maximum(W2_β), maximum(W2_T), mean(W2_β), mean(W2_T)))
            push!(outputs_real, out)

        end
        println(results_real)

        N <= 10000 && FileIO.save(joinpath(path_to_folder, "results.jld2"), "results_sim", results_sim, "outputs_sim", outputs_sim, "results_real", results_real, "outputs_real", outputs_real)
        N >= 50000 && FileIO.save(joinpath(path_to_folder, "results_largeN.jld2"), "results_sim", results_sim, "results_real", results_real, "outputs_sim", outputs_sim, "outputs_real", outputs_real)
    end
end


results_sim, _, results_real, _ = load(joinpath(path_to_folder, "results.jld2"), "results_sim", "outputs_sim", "results_real", "outputs_real")
results_sim_large, results_real_large = load(joinpath(path_to_folder, "results_largeN.jld2"), "results_sim", "results_real")


iters_sim = ones(2*length(results_sim.iters));      iters_sim[1:2:end] .= results_sim.iters;    iters_sim[2:2:end] .= results_sim_large.iters;      iters_sim = round.(Int64, iters_sim);
size_sim = ones(2*length(results_sim.iters));       size_sim[1:2:end] .= results_sim.size;      size_sim[2:2:end] .= results_sim_large.size;        size_sim = round.(Int64, size_sim);
iters_real = ones(2*length(results_real.iters));    iters_real[1:2:end] .= results_real.iters;  iters_real[2:2:end] .= results_real_large.iters;    iters_real = round.(Int64, iters_real);
size_real = ones(2*length(results_real.size));      size_real[1:2:end] .= results_real.size;    size_real[2:2:end] .= results_real_large.size;      size_real = round.(Int64, size_real);
cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]

pretty_df = DataFrame(
    Case = vcat([[case, ""] for case in cases]...),
    Real = ["$(iters) ($(size))" for (iters, size) in zip(iters_real, size_real)],
    Simulated = ["$(iters) ($(size))" for (iters, size) in zip(iters_sim, size_sim)]
)

# open(path_to_folder*"results.tex", "w") do f
#     pretty_table(f, pretty_df, backend=Val(:latex), alignment=[:l, :c, :c], hlines = 3:2:11, tf = tf_latex_double) # (v, i, j) -> round(v, digits = 5)
# end

CSV.write(joinpath(path_to_folder, "..", "summary_GG.csv"), pretty_df)


time_cg_sim = ones(2*length(results_sim.time_cg));      time_cg_sim[1:2:end] .= results_sim.time_cg;    time_cg_sim[2:2:end] .= results_sim_large.time_cg;            
time_chol_sim = ones(2*length(results_sim.time_cg));       time_chol_sim[1:2:end] .= results_sim.time_chol;      time_chol_sim[2:2:end] .= results_sim_large.time_chol;     
time_cg_real = ones(2*length(results_real.time_cg));    time_cg_real[1:2:end] .= results_real.time_cg;  time_cg_real[2:2:end] .= results_real_large.time_cg;            
time_chol_real = ones(2*length(results_real.time_cg));      time_chol_real[1:2:end] .= results_real.time_chol;    time_chol_real[2:2:end] .= results_real_large.time_chol;   
cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]

using Printf
round1(x) = @sprintf("%.2e", x)
round2(x) = @sprintf("%.2f", x)

time_df = DataFrame(
    Case = vcat([[case, ""] for case in cases]...),
    Real = ["$(time_chol/time_cg|>round2) ($(time_cg|>round1))" for (time_cg, time_chol) in zip(time_cg_real, time_chol_real)],
    Simulated = ["$(time_chol/time_cg|>round2) ($(time_cg|>round1))" for (time_cg, time_chol) in zip(time_cg_sim, time_chol_sim)]
)

CSV.write(joinpath(path_to_folder, "..", "time_GG.csv"), time_df)


cost_cg_sim = ones(2*length(results_sim.cost_cg));      cost_cg_sim[1:2:end] .= results_sim.cost_cg;    cost_cg_sim[2:2:end] .= results_sim_large.cost_cg;            
cost_chol_sim = ones(2*length(results_sim.cost_cg));       cost_chol_sim[1:2:end] .= results_sim.cost_chol;      cost_chol_sim[2:2:end] .= results_sim_large.cost_chol;     
cost_cg_real = ones(2*length(results_real.cost_cg));    cost_cg_real[1:2:end] .= results_real.cost_cg;  cost_cg_real[2:2:end] .= results_real_large.cost_cg;            
cost_chol_real = ones(2*length(results_real.cost_cg));      cost_chol_real[1:2:end] .= results_real.cost_chol;    cost_chol_real[2:2:end] .= results_real_large.cost_chol;   
cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]

using Printf
round1(x) = @sprintf("%.2e", x)
round2(x) = @sprintf("%.2f", x)

cost_df = DataFrame(
    Case = vcat([[case, ""] for case in cases]...),
    Real = ["$(time_chol/time_cg|>round2) ($(time_cg|>round1))" for (time_cg, time_chol) in zip(cost_cg_real, cost_chol_real)],
    Simulated = ["$(time_chol/time_cg|>round2) ($(time_cg|>round1))" for (time_cg, time_chol) in zip(cost_cg_sim, cost_chol_sim)]
)

CSV.write(joinpath(path_to_folder, "..", "cost_GG.csv"), cost_df)