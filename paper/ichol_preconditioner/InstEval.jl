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


df = DataFrame(CSV.File(path_to_data*"insteval.csv", delim=","))[:, 2:end]; df.y .-=1;


# Dataframes
function get_df_sel(df::DataFrame, N::Int)
    len, _ = size(df)
    inserted = zeros(Bool, len)
    queues = Queue{Int64}[]; stacks = Stack{Int64}[];
    for s in unique(df.s)
        q = Queue{Int}()
        for pos in findall(df.s .== s)
            if inserted[pos] == 0
                inserted[pos] = 1
                enqueue!(q, pos)
            end
        end
        push!(queues, q)
    end
    inserted = zeros(Bool, len)
    for d in unique(df.d)
        s = Stack{Int}()
        for pos in findall(df.d .== d)
            if inserted[pos] == 0
                inserted[pos] = 1
                push!(s, pos)
            end
        end
        push!(stacks, s)
    end

    selection = Int[];
    while length(selection) < N
        for q in queues
            isempty(q) && continue
            push!(selection, dequeue!(q))
        end
        for s in stacks
            isempty(s) && continue
            push!(selection, pop!(s))
        end
        selection = unique(selection)
    end
    return selection
end

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
    if graph_type == "ER"
        # ER random graph
        I = [2972, 1128, 4, 6, 14, 2]; 
        aux = ER_factors(N, I); y = zeros(Int64, N);
        
        # Enforcing presence of each factor to have fixed p
        aux[1:I[1], 1] = sample(1:I[1], I[1], replace=false)
        aux[1:I[2], 2] = sample(1:I[2], I[2], replace=false)


        df_sim = DataFrame(
            :y => y,
            :s => aux[:, 1], :d => aux[:, 2], :studage => aux[:, 3], :lectage => aux[:, 4], :dept => aux[:, 5], :service =>aux[:, 6]
        );
    else
        # Sample each factor independently
        F1 = vcat(sample(1:2972, N-2972), sample(1:2972, 2972, replace=false)); F2 = vcat(sample(1:1128, N-1128), sample(1:1128, 1128, replace=false));
        F3 = sample(1:4, N); F4 = sample(1:6, N); F5 = sample(1:14, N); F6 = sample(1:2, N); 
        y = zeros(Int64, N);

        df_sim = DataFrame(
            :y => y,
            :s => F1, :d => F2, :studage => F3, :lectage => F4, :dept => F5, :service =>F6
        );
    end
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
    @formula(y ~ 1 + (1 | s) + (1 | d)),
    
    # Adding nested effect (dept into d)
    @formula(y ~ 1 + (1 | s) + (1| d) + (1|dept)),

    # Adding random slopes (no nested)
    @formula(y ~ 1 + (1 | s) + (1+service| d)),

    # Adding random slopes (with nested)
    @formula(y ~ 1 + (1 | s) + (1+service| d) + (1|dept)),

    # 2 way interactions
    @formula(y ~ 1 + (1 | s) + (1 | d) + (1 | lectage) + (1 | s&d) + (1 | s&lectage) + (1 | d&lectage)),
    
    # 3 way interactions
    @formula(y ~ 1 + (1 | s) + (1 | d) + (1 | lectage) + (1 | s&d) + (1 | s&lectage) + (1 | d&lectage) + (1 | d&s&lectage)),

    # Everything
    @formula(y ~ 1 + (1 | s) + (1+service| d) + (1|dept) + (1 | lectage) + (1 | s&d) + (1 | s&lectage) + (1 | d&lectage) + (1 | d&s&lectage))
];


if false
    for N in [70000]
        results_sim  = DataFrame(iters_jacobi = Float64[], iters_ichol = Float64[], size = Int[]);
        results_real = DataFrame(iters_jacobi = Float64[], iters_ichol = Float64[], size = Int[]);
        outputs_sim = []; outputs_real = [];
        df_sim, df_real = get_dataframes(df, N)

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

        FileIO.save(path_to_folder*"results_IE.jld2", "results_sim", results_sim, "results_real", results_real, "outputs_sim", outputs_sim, "outputs_real", outputs_real)
    end
end

results_sim, outputs_sim, results_real, outputs_real = load(path_to_folder*"results_IE.jld2", "results_sim", "outputs_sim", "results_real", "outputs_real");


jacobi_sim = round.(Int64, results_sim.iters_jacobi); ichol_sim = round.(Int64, results_sim.iters_ichol); size_sim = results_sim.size
jacobi_real = round.(Int64, results_real.iters_jacobi); ichol_real = round.(Int64, results_real.iters_ichol); size_real = results_real.size

cases = ["Random intercepts", "Nested effect", "Random slopes", "Nested random slopes", "2 way interactions", "3 way interactions", "Everything"]

pretty_df = DataFrame(
    Case = cases,
    Real_jac = jacobi_real,
    Real_ichol = ichol_real,
    Real_size = size_real,
    Sim_jac = jacobi_sim,
    Sim_ichol = ichol_sim,
    Sim_size = size_sim,
)

CSV.write(path_to_folder*"summary_IE.csv", pretty_df)