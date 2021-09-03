"""
    Ambiguity Set
"""
struct Sample
    ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}    # sampled scenario
    p::Float64                                                  # associated probability
end

abstract type AbstractAmbiguitySet end

struct WassersteinSet <: AbstractAmbiguitySet
    samples::Array{Sample}  # empirical distribution
    N::Int                  # number of distinct samples
    ϵ::Float64              # radius of Wasserstein Ball
    norm_func::Function     # function that determines the norm
    WassersteinSet(samples::Array{Sample}, ϵ::Float64, norm_func::Function) = new(samples, length(samples), ϵ, norm_func)
end

function norm_L1(x::Array{Float64}, y::Array{Float64})::Float64
    val = 0
    for i in 1:length(x)
        val += abs(x[i] - y[i])
    end
    return val
end
