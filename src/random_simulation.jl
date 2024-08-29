using Random, Distributions
using LinearAlgebra
using SparseArrays 


_build_precision(V::SparseMatrixCSC{Tv,Ti}, U::SparseMatrixCSC{Tv,Ti}, τ::Tv) where{Ti<:Integer, Tv} = V + τ.*U
#

function _build_precision_and_preconditioned_precision(V::SparseMatrixCSC{Tv,Ti}, U::SparseMatrixCSC{Tv,Ti}, τ::Tv) where{Ti<:Integer, Tv} 
    Q = V + τ.*U

    norm_Q = _normalize_by_diagonal(Q)

    return Q, norm_Q
end#

function _build_V(G::Vector{R}; precisions=Dict("T_0" => 0.0, "T" => nothing), Tv=Float64, Ti=Int64) where{R<:Integer}
    K = length(G)

    T_0 = get(precisions, "T_0", -1)
    T = get(precisions, "T", -1)
    (T===nothing) && (T = ones(Tv, K))
    
    @assert length(T) == K

    # build V
    diag_V = Vector{Tv}(undef, sum(G)+1)
    index = 0
    for (i, t) in zip(G, T)
        diag_V[(index+1):(index+i)] .= t
        index += i
    end
    diag_V[end] = T_0

    return spdiagm(diag_V)
end #

# ER K-partite graph
function _remove_unobserved_levels(Z::SparseMatrixCSC{Tv, Ti}, G::Vector{R}; verbose::Bool=false) where{Tv, Ti, R}
    # Needs the G(Z, dims=2).>0)

    ptr = cumsum(vcat(1, G))
    G .= eltype(G)[sum(sel[ptr[i]:ptr[i+1]-1]) for i in 1:length(G)] # updated G
    verbose && println("Removed $(size(Z, 1)-sum(sel)) out of $(size(Z, 1)) levels")
    return Z[sel, :] 
end #

function _generate_ER_U(N::Integer, G::Vector{R}; kwargs...) where {R<:Integer}
    Z = _generate_ER_Z(N, G; kwargs...)
    return Z * transpose(Z)
end #

function _generate_ER_U(G::Vector{R}, π::AbstractFloat; kwargs...) where{R<:Integer}
    K_dim_array_size = prod(G)
    N = rand(Binomial(K_dim_array_size, π))

    Z = _generate_ER_Z(N, G; kwargs...)
    return Z * transpose(Z)
end #

function _generate_ER_Z(N::Integer, G::Vector{R}; remove_unobserved_levels::Bool=false, Tv=Float64, Ti=Int64, seed=nothing, debug::Bool=false, ordered::Bool=false) where{R<:Integer}
    if seed !== nothing
        Random.seed!()
    end
    
    p = sum(G) + 1
    K = length(G) 
    K_dim_array_size = prod(G) 

    Z_colptr = [1+i*(K+1) for i in range(0, N)]
    Z_rowval = Array{Ti}(undef, (K+1)*N)
    Z_nzval = ones(Tv, length(Z_rowval))

    samples_from_mult_array = sample(1:K_dim_array_size, N, replace=false, ordered=ordered)

    cartesian = CartesianIndices(Tuple(G))

    for (sample, ptr) in zip(samples_from_mult_array, Z_colptr[1:(end-1)])
        levels = collect(Tuple(cartesian[sample]))
        Z_rowval[ptr: (ptr+K)] .= convert(Vector{Ti}, cumsum(vcat(0, G)).+ vcat(levels, 1))

        debug && println(levels)
    end

    Z = SparseMatrixCSC{Tv, Ti}(p, N, Z_colptr, Z_rowval, Z_nzval)
    remove_unobserved_levels && return _remove_unobserved_levels(Z, G)
    return Z
end #