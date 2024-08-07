import LinearAlgebra.ldiv!
using LinearAlgebra
using SparseArrays
using SuiteSparse


struct JacobiPreconditioner{Tv<:Union{Float64}}
    Pl::Vector{Tv}

    JacobiPreconditioner(Q::SparseMatrixCSC{Tv, Ti}) where{Tv, Ti<:Integer} = new{Tv}(Vector(1 ./ diag(Q)))
end

function ldiv!(x::AbstractArray, F::JacobiPreconditioner, b::AbstractArray)
    @inbounds @simd for i in eachindex(b)
        x[i] = F.Pl[i]*b[i]
    end
end


mutable struct ICholPreconditioner{Tv<:Union{Float64}} 
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    L::LowerTriangular{Tv, SparseMatrixCSC{Tv, Int64}}
    L_T::UpperTriangular{Tv, SparseMatrixCSC{Tv, Int64}}
    failed::Bool

    function ICholPreconditioner(Q::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti<:Integer}
        L = tril(Q)
        L_T = triu(Q)
        fill!(L.nzval, 0.)
        fill!(L_T.nzval, 0.)
        failed = false
        @inbounds for col in 1:Q.n
            Q_jj = Q[col, col]
            
            temp = Q_jj - sum(abs2, L_T.nzval[L_T.colptr[col]: (L_T.colptr[col+1]-1)])
            if temp<0
                failed = true
            end
            
            L_jj = sqrt(abs(temp))
            L[col, col] = L_jj
            L_T[col, col] = L_jj
            
            from = L.colptr[col]
            to = L.colptr[col+1] - 1
    
            @inbounds for row in L.rowval[(from+1): to]
                Q_ij = Q[row, col]
                aux = Q_ij - SparseArrays._spdot(dot,
                                                        L_T.colptr[col], L_T.colptr[col+1]-1, L_T.rowval, L_T.nzval,
                                                        L_T.colptr[row], L_T.colptr[row+1]-1, L_T.rowval, L_T.nzval
                )

                L[row, col] = aux / L_jj
                L_T[col, row] = aux / L_jj
            end
        end

        return new{eltype(Q)}(Q.m, Q.n, L, L_T, failed)
    end
end

function ldiv!(x::AbstractArray, F::ICholPreconditioner, b::AbstractArray)
    ldiv!(x, F.L, b)
    ldiv!(x, F.L_T, x)
end