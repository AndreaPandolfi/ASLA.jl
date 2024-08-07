using Random
using Distributions, PolyaGammaDistribution
using MixedModels, StatsModels
using DataFrames
using Tables
using ProgressMeter

_get_D(form::FormulaTerm) = [length(rand_eff.lhs.terms) for rand_eff in form.rhs[2:end]]

_get_G(Xs::Tuple, D::Vector{Int64}) = [size(x)[2] for x in Xs[2:end]] .÷ D

function PGgibbs_GLMMs(df::DataFrame, f::FormulaTerm, n_iter::Integer;
    hyperparams=nothing,
    initial_values=nothing,
    system_solver!::Function = function (x, Q, b) x.=Q\b; return nothing end, 
    seed=nothing, burn_in::Integer=0,
    debug=false, converged_values=false
    )
    @assert burn_in < n_iter

    if seed!==nothing
        Random.seed!(seed)
    end
    
    tbl = Tables.columntable(df)
    form = MixedModels.schematize(f, tbl, Dict{Symbol,Any}())
    y, Xs = MixedModels.modelcols(form, tbl)
    n = length(unique(y))-1 # where y ~ Binom(n, x' β)

    V = hcat(sparse.(Xs[2:end])..., Xs[1]) # fixed effects in last position
    # V_T = spzeros(size(V)[2], size(V)[1]); V_T = transpose!(V_T, V)
    V_T = transpose(V)
    y_centered = y .- 0.5*n

    G_0 = size(Xs[1])[2] # fixed-effects dimension
    N = length(y)
    D = _get_D(form)
    G = _get_G(Xs, D)
    K = length(G)
    p = sum([size(x)[2] for x in Xs]) # fixed-effs + rand-effs dimension

    if hyperparams===nothing
        HPS = Dict(
            :WiDf => 0.1.*ones(K),
            :WiInvScale => [if (D_k==1) 0.1 else Matrix(0.1*LinearAlgebra.I, D_k, D_k) end for D_k in D]
        )
    else
        HPS = hyperparams # short for hyperparams
    end

    if initial_values===nothing
        initial_values = Dict(
        :θ => zeros(p), # [random effects, fixed effects]
        :T  => [Matrix(1.0*LinearAlgebra.I, D_k, D_k) for D_k in D],
        :Ω => ones(N)        
        )
    end
    θ = deepcopy(initial_values[:θ])
    T = deepcopy(initial_values[:T])
    Ω = deepcopy(initial_values[:Ω])
    
    β_hist = typeof(θ)[]; push!(β_hist, θ[end-G_0 + 1:end])
    T_hist = typeof(T)[];   push!(T_hist, T)

    debug && (full_hist = Dict(
        :Ω => typeof(Ω)[],
        :T => typeof(T)[],
        :θ => typeof(θ)[]
    ))

    @showprogress for _ in 1:n_iter
        # update θ
        prior_prec = blockdiag(vcat([repeat([sparse(T_k)], G_k) for (T_k, G_k) in zip(T, G)]...)..., spzeros(G_0, G_0))
        T_sqrt = [cholesky(Symmetric(T_k), NoPivot()).L for T_k in T]
        prior_prec_sqrt = blockdiag(vcat([repeat([sparse(L_k)], G_k) for (L_k, G_k) in zip(T_sqrt, G)]...)..., spzeros(G_0, G_0))
        
        Q = prior_prec + V_T*spdiagm(Ω)*V
        b = V_T*y_centered + V_T * (sqrt.(Ω) .* rand(Normal(0, 1), N)) + prior_prec_sqrt*rand(Normal(0,1), p) 
        system_solver!(θ, Q, b)
        # F = cholesky(Symmetric(Q), NoPivot())
        # w = F.L\(V_T*y_centered)
        # θ = F.U\(w + rand(Normal(0,1), p))
        
        # debug && println(θ[1:3])

        push!(β_hist, θ[end-G_0 + 1:end])

        # update Ω
        Ω = rand.(PolyaGamma.(n,V*θ))

        # update T
        lengths = [size(x)[2] for x in Xs[2:end]]
        ends = cumsum(lengths)
        start = 1; End = lengths[1]
        for (k, l, End) in zip(1:K, lengths, ends)
            # println("$(l), $(start), $(End)")
            if D[k]==1
                aux = sum(θ[start:End].^2)
                T[k] = [rand(Gamma(HPS[:WiDf][k] + G[k]/2, 1/(HPS[:WiInvScale][k] + aux/2)));;] # Gamma is parametrized with shape and SCALE!!
            else
                aux = sum([x*transpose(x) for x in Base.Iterators.partition(θ[start:End], D[k])])
                T[k] = rand(Wishart(HPS[:WiDf][k] + G[k], Matrix(inv(Symmetric(HPS[:WiInvScale][k] + aux)))))
            end
            start += l
        end
        
        push!(T_hist, copy(T))

        if debug
            push!(full_hist[:Ω], Ω)
            push!(full_hist[:T], T)
            push!(full_hist[:θ], θ)
        end
    end
    debug && return full_hist, T_hist
    converged_values && return Dict(:θ => deepcopy(θ), :T  => deepcopy(T), :Ω => deepcopy(Ω))
    return β_hist[burn_in+1:end], T_hist[burn_in+1:end]
end