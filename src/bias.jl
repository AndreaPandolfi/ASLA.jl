using StatsBase

# Empirical estimate of L1 and Wasserstein-2 norm, for univariate empirical distributions

function L1_sample_distance(x::Vector{S}, y::Vector{S}) where{S<:AbstractFloat}
    @assert length(x)>=1000 && length(y)>=1000 "Need larger sample size"
    u = maximum([maximum(x), maximum(y)])
    l = minimum([minimum(x), minimum(y)])
    n_bins = length(x)÷50
    bins = range(l, u, length=n_bins)
    
    fx = normalize(fit(Histogram, x, bins), mode=:pdf)
    fy = normalize(fit(Histogram, y, bins), mode=:pdf)

    sum(abs.(fx.weights - fy.weights))*(bins[end]-bins[end-1])
end

# From https://www.stat.cmu.edu/~larry/=sml/Opt.pdf page 5
function W2_sample_distance(x::Vector{S}, y::Vector{S}) where{S<:AbstractFloat}
    @assert length(x) == length(y) "Need samples of the same length"

    sqrt(sum((sort(x) - sort(y)).^2))
end

function W2_sample_distance(x::Vector{S}, y::Vector{S}) where{S<:Union{Matrix, Vector}}
    @assert length(x) == length(y) "Need samples of the same length"

    elementwise_W2 = copy(x[1])
    for index in eachindex(elementwise_W2)
        elementwise_W2[index] = W2_sample_distance([el[index] for el in x], [el[index] for el in y])
    end
    return elementwise_W2
end