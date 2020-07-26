module DRMSMIP

using JuMP, Gurobi
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
struct TreeNode
    parent::Int                                 # index of parent node
    children::Vector{Int}                       # indices of child nodes
    k::Int                                      # current stage
    ξ::Vector{Float64}                          # current scenario
    set::Union{AbstractAmbiguitySet, Nothing}   # ambiguity set for child nodes
    cost::Vector{Float64}
end
mutable struct Tree
    nodes::Vector{TreeNode}     # list of nodes
    K::Int                      # length of tree
end

Tree(ξ::Vector{Float64}, set::AbstractAmbiguitySet, cost::Vector{Float64}) = 
    Tree([TreeNode(0, Vector{Int}(), 1, ξ, set, cost )], 1)

function addchild!(tree::Tree, id::Int, ξ::Vector{Float64},
             set::Union{AbstractAmbiguitySet, Nothing}, cost::Vector{Float64})
    #   adds child node to tree.nodes[id]
    1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))   # check if id is valid
    k = get_stage(tree, id) + 1                                     # get new stage value
    push!(tree.nodes, TreeNode(id, Vector{}(), k, ξ, set, cost ))   # push to node list
    child_id = length(tree.nodes)                                   # get current node ID
    push!(tree.nodes[id].children, child_id)                        # push child_id to parent node children
    if k > tree.K
        tree.K = k  # update length of tree to the maximum value
    end
end

get_children(tree, id) = tree.nodes[id].children
get_parent(tree,id) = tree.nodes[id].parent
get_stage(tree, id) = tree.nodes[id].k
get_scenario(tree, id) = tree.nodes[id].ξ

function get_history(tree::Tree, id::Int)::Array{Int}
    # gets a vector of tree node IDs up until current
    stage = get_stage(tree, id)
    hist = Array{Int}(undef, stage)

    current_id = id
    for k = stage:-1:1
        hist[k] = current_id
        current_id = get_parent(tree, current_id)
    end
    return hist
end

function get_future(tree::Tree, root_id::Int)::Array{Int}
    #   output list of all leaf node IDs branching from root_id
    arr_leaves = Int[]

    function iterate_children(tree::Tree, id::Int)
        children = get_children(tree, id)
        if length(children) == 0
            #buffer output
            push!(arr_leaves, id)
        else
            for child in children
                iterate_children(tree, child)
            end
        end
    end

    iterate_children(tree, root_id)
    return arr_leaves
end

function get_stage_id(tree::Tree)::Array{Array{Int}}
    # gets a list of tree node IDs separated by stages
    K = tree.K
    nodelist = [ Int[] for _ in 1:K]

    for id in 1:length(tree.nodes)
        k = get_stage(tree, id)
        push!(nodelist[k], id)
    end
    return nodelist
end

include("BlockModel.jl")
include("LagrangeDual.jl")

end