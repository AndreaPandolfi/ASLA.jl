using ASLA: log10_range, _build_V, _build_precision, _generate_ER_U, cost_per_iter_CG, cost_cholesky, log_scatter_and_linear_approximation, log_scatter_and_linear_approximation!

using Random, Distributions
using LinearAlgebra, SparseArrays, IterativeSolvers
using JLD2, FileIO
using Plots

path_to_folder = "paper/cholesky_cg_runtime/"

function generate_Q(I::Vector{T}, π::AbstractFloat, τ) where{T<:Integer}
    K_dim_array_size = prod(I)
    N = rand(Binomial(K_dim_array_size, π))
    
    return N, _build_precision(_build_V(I), _generate_ER_U(N, I), τ)
end


# function symrcm(A::SparseMatrixCSC, degrees::Vector{T}; sortbydeg=false) where{T<:Integer}
#     ag = adjgraph(A; sortbydeg = sortbydeg)
#     return symrcm(ag, degrees)
# end

function compute_costs(Q::SparseMatrixCSC; K=2)
    L = sparse(cholesky(Q).L)
    # L_AMD = sparse(cholesky(Q).L)
    # L_MMD = sparse(cholesky(Matrix(Q), RowMaximum()).L)
    # RCM_perm = symrcm(Q, Vector{Int64}(K*diag(Q)))
    # L_RCM = sparse(cholesky(Q, perm=RCM_perm).L)
    
    b = rand(Q.m) .- 0.5
    P = Diagonal(diag(Q))
    _, ch = cg(Q, b, Pl = P, log=true)
    cost_cg = ch.iters*cost_per_iter_CG(Q)

    # return cost_cholesky(L), cost_cholesky(L_AMD), cost_cholesky(L_RCM), ch.iters, cost_cg
    return cost_cholesky(L), ch.iters, cost_cg
    
end

data1 = false; data2 = false; data3 = true
plot1 = true; plot2 = true; plot3 = true

# ------------------ PLOT 1: K = 2, π = 20/I ------------------ 
if data1
    cost_L = []; cost_cg = []; iters_cg = []; # cost_L_AMD = []; cost_L_RCM = []; 
    Is = floor.(Int64, log10_range(log10(50), log10(2000), 6))
    Ns = eltype(Is)[]
    for I in Is
        println("I = $(I)")
        N = []; cL = []; iters = []; cg = [];
        for _ in 1:30
            _N, _Q = generate_Q(repeat([I], 2), 20/I, 1.0)
            _cL, _iters, _cg = compute_costs(_Q)
            push!(N, _N); push!(cL, _cL); push!(iters, _iters); push!(cg, _cg)
        end
        push!(Ns,       round(Int64, mean(N)))
        push!(cost_L,   round(Int64, mean(cL)))
        push!(iters_cg, round(Int64, mean(iters)))
        push!(cost_cg,  round(Int64, mean(cg)))
    end

    dict_plot1 = Dict(:Is => Is, :Ns => Ns, :L => cost_L, :cg => cost_cg, :iters_cg => iters_cg) # :L_AMD => cost_L_AMD, :L_RCM => cost_L_RCM)
    FileIO.save(path_to_folder*"cholesky_runtime1.jld2", "dict_plot1", dict_plot1)
end

if plot1
    D1 = FileIO.load(path_to_folder*"cholesky_runtime1.jld2", "dict_plot1")

    log_scatter_and_linear_approximation(
        D1[:Is]*2, D1[:L], linecolor=:red, 
            markersize=3, markercolor=:red, markershape=:circle, 
            legend_position=:topleft,
            label="Cholesky",
            size=(300,300))
    # log_scatter_and_linear_approximation!(D1[:Is], D1[:L_AMD], linecolor=:green, markersize=3, markercolor=:green, markershape=:utriangle, label="Chol-AMD")
    # log_scatter_and_linear_approximation!(D1[:Is], D1[:L_RCM], linecolor=:blue, markersize=3, markercolor=:blue, markershape=:rect, label="Chol-RCM")
    log_scatter_and_linear_approximation!(D1[:Is]*2, D1[:cg], linecolor=:purple, markersize=3, markercolor=:purple, markershape=:utriangle, label="CG")
    xticks!([100, 300, 1000, 3000], ["100", "300", "1000", "3000"])
    yticks!(10 .^ (4:2:11))
    xlims!((80, 4600))
    ylims!((10^4, 10^10.3))
    ylabel!("flops")
    xlabel!("p")
    savefig(path_to_folder*"Erdos_K2_d20_Cost.pdf")
end

# ------------------ PLOT 2: K = 2, π = sqrt(I) ------------------ 
if data2
    cost_L = []; cost_cg = []; iters_cg = []; # cost_L_AMD = []; cost_L_RCM = []; 
    Is = floor.(Int64, log10_range(log10(50), log10(2000), 6))
    Ns = eltype(Is)[]
    for I in Is
        println("I = $(I)")
        N = []; cL = []; iters = []; cg = [];
        for _ in 1:30
            _N, _Q = generate_Q(repeat([I], 2), 1/sqrt(I), 1.0)
            _cL, _iters, _cg = compute_costs(_Q)
            push!(N, _N); push!(cL, _cL); push!(iters, _iters); push!(cg, _cg)
        end
        push!(Ns,       round(Int64, mean(N)))
        push!(cost_L,   round(Int64, mean(cL)))
        push!(iters_cg, round(Int64, mean(iters)))
        push!(cost_cg,  round(Int64, mean(cg)))
    end

    dict_plot2 = Dict(:Is => Is, :Ns => Ns, :L => cost_L, :cg => cost_cg, :iters_cg => iters_cg) # :L_AMD => cost_L_AMD, :L_RCM => cost_L_RCM)
    FileIO.save(path_to_folder*"cholesky_runtime2.jld2", "dict_plot2", dict_plot2)
end

if plot2
    D2 = FileIO.load(path_to_folder*"cholesky_runtime2.jld2", "dict_plot2")

    log_scatter_and_linear_approximation(
            D2[:Is]*2, D2[:L], linecolor=:red, 
            markersize=3, markercolor=:red, markershape=:circle, 
            legend_position=:topleft,
            # legend = false,
            label="",
            size=(300,300))
    # log_scatter_and_linear_approximation!(D2[:Is], D2[:L_AMD], linecolor=:green, markersize=3, markercolor=:green, markershape=:utriangle, label="Chol-AMD")
    # log_scatter_and_linear_approximation!(D2[:Is], D2[:L_RCM], linecolor=:blue, markersize=3, markercolor=:blue, markershape=:rect, label="Chol-RCM")
    log_scatter_and_linear_approximation!(D2[:Is]*2, D2[:cg], linecolor=:purple, markersize=3, markercolor=:purple, markershape=:utriangle, label="")
    xticks!([100, 300, 1000, 3000], ["100", "300", "1000", "3000"])
    yticks!(10 .^ (4:2:11))
    xlims!((80, 4600))
    ylims!((10^4, 10^10.3))
    xlabel!("p")
    savefig(path_to_folder* "Erdos_K2_dsqrtI_Cost.pdf")
end


# ------------------ PLOT 3: K = 5, π = I^(-K + 3/2) ------------------ 
if data3
    cost_L = []; cost_cg = []; iters_cg = []; # cost_L_AMD = []; cost_L_RCM = []; 
    # Is = floor.(Int64, log10_range(log10(50), log10(800), 6))
    Is = floor.(Int64, log10_range(log10(50), log10(2000), 6) .* (2//5))
    Ns = eltype(Is)[]
    K = 5
    for I in Is
        println("I = $(I)")
        N = []; cL = []; iters = []; cg = [];
        for _ in 1:30
            π = I ^(-K + 3/2)
            _N, _Q = generate_Q(repeat([I], K), π, 1.0)
            _cL, _iters, _cg = compute_costs(_Q)
            push!(N, _N); push!(cL, _cL); push!(iters, _iters); push!(cg, _cg)
        end
        push!(Ns,       round(Int64, mean(N)))
        push!(cost_L,   round(Int64, mean(cL)))
        push!(iters_cg, round(Int64, mean(iters)))
        push!(cost_cg,  round(Int64, mean(cg)))
    end

    dict_plot3 = Dict(:Is => Is, :Ns => Ns, :L => cost_L, :cg => cost_cg, :iters_cg => iters_cg) # :L_AMD => cost_L_AMD, :L_RCM => cost_L_RCM)
    FileIO.save(path_to_folder*"cholesky_runtime3.jld2", "dict_plot3", dict_plot3)
end

if plot3
    D3 = FileIO.load(path_to_folder*"cholesky_runtime3.jld2", "dict_plot3")

    log_scatter_and_linear_approximation(
        D3[:Is]*5, D3[:L], linecolor=:red, 
            markersize=3, markercolor=:red, markershape=:circle, 
            legend_position=:topleft,
            label="",
            size=(300,300))
    # log_scatter_and_linear_approximation!(D3[:Is], D3[:L_AMD], linecolor=:green, markersize=3, markercolor=:green, markershape=:utriangle, label="Chol-AMD")
    # log_scatter_and_linear_approximation!(D3[:Is], D3[:L_RCM], linecolor=:blue, markersize=3, markercolor=:blue, markershape=:rect, label="Chol-RCM")
    log_scatter_and_linear_approximation!(D3[:Is]*5, D3[:cg], linecolor=:purple, markersize=3, markercolor=:purple, markershape=:utriangle, label="")
    xticks!([100, 300, 1000, 3000], ["100", "300", "1000", "3000"])
    yticks!(10 .^ (4:2:11))
    xlims!((80, 4600))
    ylims!((10^4, 10^10.3))
    xlabel!("p")
    savefig(path_to_folder*"Erdos_K5_dsqrtI_Cost.pdf")
end



