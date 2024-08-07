using Plots, StatsPlots
using DataFrames
using GLM

# computational-cost functions
cost_per_iter_CG(Q::SparseMatrixCSC{Tv, Ti}) where{Tv, Ti<:Integer} = 10*Q.m + length(Q.nzval)

function cost_cholesky(L::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti<:Integer}
    n_L = L.colptr[2:end] - L.colptr[1:end-1]
    return sum((n_L .+ 1).^2)
end

# normalize matrices
function _normalize_by_diagonal(Q::Union{Matrix{Tv}, SparseMatrixCSC{Tv,Ti}}) where{Tv, Ti}
    function inv_sqrt(x)
        @assert x>=0  
        x==zero(eltype(x)) && return zero(eltype(x))
        return 1/sqrt(x)
    end

    D = spdiagm(inv_sqrt.(diag(Q)))
    return D * Q * D
end

function _normalize_adjacency(A::Union{Matrix{Tv}, SparseMatrixCSC{Tv,Ti}}) where{Tv, Ti}
    function inv_sqrt(x)
        @assert x>=0  
        x==zero(eltype(x)) && return zero(eltype(x))
        return 1/sqrt(x)
    end

    D = spdiagm(inv_sqrt.(vec(sum(A, dims=2))))
    return D * A * D
end

# log ranges
log10_range(start,stop,length::Integer) = 10 .^ (range(start, stop, length=length))
ln_range(start,stop,length::Integer) = exp.(range(start, stop, length=length))

# Plotting
function scatter_and_linear_approximation(x,y; linecolor=:black, kwargs...)
    ols = lm(@formula(Y ~ X), DataFrame(X=x, Y=y))
    slope = coef(ols)[2]
    intercept = coef(ols)[1]
    scatter(x, y; xaxis=:log, yaxis=:log, kwargs...)
    plot!(x, intercept .+ x.*slope, color=linecolor, label="Slope: $(round(slope, digits=3))", linestyle=:dash)
end

function log_scatter_and_linear_approximation(x,y; linecolor=:black, kwargs...)
    ols = lm(@formula(Y ~ X), DataFrame(X=log.(x), Y=log.(y)))
    slope = coef(ols)[2]
    intercept = coef(ols)[1]
    scatter(x, y; xaxis=:log, yaxis=:log, kwargs...)
    plot!(x, exp(intercept).* x.^slope, color=linecolor, label="Slope: $(round(slope, digits=3))", linestyle=:dash)
end

function log_scatter_and_linear_approximation!(x,y; linecolor=:black, kwargs...)
    ols = lm(@formula(Y ~ X), DataFrame(X=log.(x), Y=log.(y)))
    slope = coef(ols)[2]
    intercept = coef(ols)[1]
    scatter!(x, y; xaxis=:log, yaxis=:log, kwargs...)
    plot!(x, exp(intercept).* x.^slope, color=linecolor, label="Slope: $(round(slope, digits=3))", linestyle=:dash)
end