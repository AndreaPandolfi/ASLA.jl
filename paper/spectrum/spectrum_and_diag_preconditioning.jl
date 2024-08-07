using ASLA: _build_V, _build_precision, _build_precision_and_preconditioned_precision, _generate_ER_U

using LinearAlgebra, SparseArrays
using Plots, LaTeXStrings

path_to_folder = "paper/spectrum/"

function generate_Q(N::Integer, I::Vector{T}; preconditioned::Bool=false, τ::AbstractFloat=1.0, kwargs...) where{T<:Integer}
    U = _generate_ER_U(N, I, kwargs...)
    V = _build_V(I, precisions=Dict("T_0" => 1.0, "T" => 1.0 .* ones(length(I))))
    
    preconditioned && return _build_precision_and_preconditioned_precision(V, U, τ)
    return _build_precision(V, U, τ)
end

I = [300, 1000, 2000]
N = 30000

Q, Q_bar = generate_Q(N, I, preconditioned=true)

eigs = eigvals(Matrix(Q))[1:(end-2)]

eigs_bar = eigvals(Matrix(Symmetric(Q_bar)))[1:(end-2)]

plot1 = histogram(eigs, label="", bins=100, normalize = :pdf)
plot1 = vline!(N./I .+ 1, linestyle=:dash, label="", linewidth=2)

plot2 = histogram(eigs_bar, label="", legend=:topleft, bins=100, normalize = :pdf)

hist = plot(plot1, plot2, size=(800, 400))

savefig(hist, path_to_folder*"spectrum_histograms.pdf")