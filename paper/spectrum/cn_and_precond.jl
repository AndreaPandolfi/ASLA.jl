using ASLA: _build_V, _build_precision, _build_precision_and_preconditioned_precision, _generate_ER_U

using  DataFrames
using LinearAlgebra, SparseArrays, IterativeSolvers

path_to_folder = "paper/spectrum/"

function compute_cn(Q, n_to_leave_out::Integer; return_eigvals::Bool=false)
    # it leaves out the "n_to_leave_out" extreamal eigenvalues on the left and right
    large_eig = IterativeSolvers.lobpcg(Q, true, 1+n_to_leave_out).λ[end]
    small_eig = IterativeSolvers.lobpcg(Q, false, 1+n_to_leave_out).λ[end]
    if return_eigvals
        return (small_eig, large_eig)
    else
        return large_eig/small_eig
    end
end

function generate_Q(N::Integer, I::Vector{T}; preconditioned::Bool=false, τ::AbstractFloat=1.0, kwargs...) where{T<:Integer}
    U = _generate_ER_U(N, I, kwargs...)
    V = _build_V(I, precisions=Dict("T_0" => 1.0, "T" => 1.0 .* ones(length(I))))
    
    preconditioned && return _build_precision_and_preconditioned_precision(V, U, τ)
    return _build_precision(V, U, τ)
end


I1 = 100
cs = [1, 3, 10, 30, 100]

norm_cn     = Vector{Float64}(undef, length(cs))
cn          = Vector{Float64}(undef, length(cs))
iters       = Vector{Int64}(undef, length(cs))
norm_iters  = Vector{Int64}(undef, length(cs)) 

for (i, c) in enumerate(cs)
    sum_I =  (1+c)*I1
    N = floor(Int64, sum_I^(3/2))
    I = [sum_I÷2, sum_I÷2]
    Q, Q_bar = generate_Q(N, I, preconditioned=true)

    κ = compute_cn(Q, 2)

    κ_bar = compute_cn(Symmetric(Q_bar), 2)

    # CONJUGATE GRADIENT
    b = rand(Float64, Q.m) .- 0.5
    P = Diagonal(diag(Q))
    _, ch       = cg(Q, b, log=true)
    _, norm_ch  = cg(Q, b, Pl = P, log=true)

    cn[i]           = κ
    norm_cn[i]      = κ_bar
    iters[i]        = ch.iters
    norm_iters[i]   = norm_ch.iters
end

df1 = DataFrame(
"I_1"                   => ((1 .+ cs) .* I1) .÷ 2,
"I_2"                   => ((1 .+ cs) .* I1) .÷ 2,
"N"                     => floor.(Int64, ((1 .+ cs) .* I1) .^ (3/2)),
"cn"                    => cn,
"n. iters"              => iters,
"cn w/ prec."           => norm_cn,
"n. iters w/ prec."     => norm_iters,
);
# store to file
open(path_to_folder*"balanced_factors.tex", "w") do f
    pretty_table(f, df2, backend=Val(:latex), alignment=[:c, :c, :c, :c, :c, :c, :c], tf = tf_latex_double) 
end



# CASE I2 = c * I1, and c = [2, 5, 10, 100]
# Random.seed!(114)

norm_cn     = Vector{Float64}(undef, length(cs))
cn          = Vector{Float64}(undef, length(cs))
iters       = Vector{Int64}(undef, length(cs))
norm_iters  = Vector{Int64}(undef, length(cs)) 
for (i, c) in enumerate(cs)
    I2 =  c*I1
    I = [I1, I2]
    sum_I =  sum(I)
    N = floor(Int64, sum_I^(1.1))

    Q, Q_bar = generate_Q(N, I, preconditioned=true)

    κ = compute_cn(Q, 2)

    κ_bar  = compute_cn(Symmetric(Q_bar), 2)

    # CONJUGATE GRADIENT
    b = rand(Float64, Q.m) .- 0.5
    P = Diagonal(diag(Q))
    _, ch       = cg(Q, b, log=true)
    _, norm_ch  = cg(Q, b, Pl = P, log=true)

    cn[i]           = κ
    norm_cn[i]      = κ_bar
    iters[i]        = ch.iters
    norm_iters[i]   = norm_ch.iters
end

df2 = DataFrame(
"I_1"                   => repeat([I1], length(cs)),
"I_2"                   => cs.*repeat([I1], length(cs)),
"N"                     => floor.(Int64, ((1 .+ cs) .* I1) .^ (3/2)),
"cn"                    => cn,
"n. iters"              => iters,
"cn w/ prec."           => norm_cn,
"n. iters w/ prec."     => norm_iters,
)

open(path_to_folder*"unbalanced_factors.tex", "w") do f
    pretty_table(f, df2, backend=Val(:latex), alignment=[:c, :c, :c, :c, :c, :c, :c], tf = tf_latex_double) 
end