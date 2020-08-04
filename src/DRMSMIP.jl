module DRMSMIP

using JuMP, Ipopt, GLPK
using DualDecomposition, BundleMethod

const DD = DualDecomposition
const BM = BundleMethod


"""
    Ambiguity Set
"""
struct Sample
    ξ::Vector{Float64}  # sampled scenario
    p::Float64          # associated probability
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

"""
    Scenario Tree
"""
struct DR_TreeNode
    parent::Int                                 # index of parent node
    children::Vector{Int}                       # indices of child nodes
    k::Int                                      # current stage
    ξ::Vector{Float64}                          # current scenario
    set::Union{AbstractAmbiguitySet, Nothing}   # ambiguity set for child nodes
    cost::Vector{Float64}
end
mutable struct DR_Tree <:DD.AbstractTree
    nodes::Vector{DR_TreeNode}     # list of nodes
    K::Int                      # length of tree
end

DR_Tree(ξ::Vector{Float64}, set::AbstractAmbiguitySet, cost::Vector{Float64}) = 
    DR_Tree([DR_TreeNode(0, Vector{Int}(), 1, ξ, set, cost )], 1)

function DD.addchild!(tree::DR_Tree, id::Int, ξ::Vector{Float64},
             set::Union{AbstractAmbiguitySet, Nothing}, cost::Vector{Float64})
    #   adds child node to tree.nodes[id]
    1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))   # check if id is valid
    k = DD.get_stage(tree, id) + 1                                     # get new stage value
    push!(tree.nodes, DR_TreeNode(id, Vector{}(), k, ξ, set, cost ))   # push to node list
    child_id = length(tree.nodes)                                   # get current node ID
    push!(tree.nodes[id].children, child_id)                        # push child_id to parent node children
    if k > tree.K
        tree.K = k  # update length of tree to the maximum value
    end
end


include("BlockModel.jl")
include("LagrangeDual.jl")

end