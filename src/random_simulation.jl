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

function _build_V(I::Vector{R}; precisions=Dict("T_0" => 0.0, "T" => nothing), Tv=Float64, Ti=Int64) where{R<:Integer}
    K = length(I)

    T_0 = get(precisions, "T_0", -1)
    T = get(precisions, "T", -1)
    (T===nothing) && (T = ones(Tv, K))
    
    @assert length(T) == K

    # build V
    diag_V = Vector{Tv}(undef, sum(I)+1)
    index = 0
    for (i, t) in zip(I, T)
        diag_V[(index+1):(index+i)] .= t
        index += i
    end
    diag_V[end] = T_0

    return spdiagm(diag_V)
end #

# ER K-partite graph
function _remove_unobserved_levels(Z::SparseMatrixCSC{Tv, Ti}, I::Vector{R}; verbose::Bool=false) where{Tv, Ti, R}
    # Needs the I::Vector, to change the n. of levels accordingly
    sel = vec(sum(Z, dims=2).>0)

    ptr = cumsum(vcat(1, I))
    I .= eltype(I)[sum(sel[ptr[i]:ptr[i+1]-1]) for i in 1:length(I)] # updated I
    verbose && println("Removed $(size(Z, 1)-sum(sel)) out of $(size(Z, 1)) levels")
    return Z[sel, :] 
end #

function _generate_ER_U(N::Integer, I::Vector{R}; kwargs...) where {R<:Integer}
    Z = _generate_ER_Z(N, I; kwargs...)
    return Z * transpose(Z)
end #

function _generate_ER_U(I::Vector{R}, π::AbstractFloat; kwargs...) where{R<:Integer}
    K_dim_array_size = prod(I)
    N = rand(Binomial(K_dim_array_size, π))

    Z = _generate_ER_Z(N, I; kwargs...)
    return Z * transpose(Z)
end #

function _generate_ER_Z(N::Integer, I::Vector{R}; remove_unobserved_levels::Bool=false, Tv=Float64, Ti=Int64, seed=nothing, debug::Bool=false) where{R<:Integer}
    if seed !== nothing
        Random.seed!()
    end
    
    p = sum(I) + 1
    K = length(I) 
    K_dim_array_size = prod(I) 

    Z_colptr = [1+i*(K+1) for i in range(0, N)]
    Z_rowval = Array{Ti}(undef, (K+1)*N)
    Z_nzval = ones(Tv, length(Z_rowval))

    samples_from_mult_array = sample(1:K_dim_array_size, N, replace=false)

    cartesian = CartesianIndices(Tuple(I))

    for (sample, ptr) in zip(samples_from_mult_array, Z_colptr[1:(end-1)])
        levels = collect(Tuple(cartesian[sample]))
        Z_rowval[ptr: (ptr+K)] .= convert(Vector{Ti}, cumsum(vcat(0, I)).+ vcat(levels, 1))

        debug && println(levels)
    end

    Z = SparseMatrixCSC{Tv, Ti}(p, N, Z_colptr, Z_rowval, Z_nzval)
    remove_unobserved_levels && return _remove_unobserved_levels(Z, I)
    return Z
end #