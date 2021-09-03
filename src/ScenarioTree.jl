"""
    DR_TreeNode

Tree node stores information of stage problem

    - `id`: index of node
    - `stage_builder`: adds variables and constraints to model with input (tree::Tree, subtree::SubTree, node::SubTreeNode)
    - `parent`: index of parent node
    - `children`: indices of child nodes
    - `stage`: current stage
    - `ξ`: current scenario
    - `set`: ambiguity set
    - 'cost': current scenario of cost
"""

mutable struct DR_TreeNode <: DD.AbstractTreeNode
    id::Int
    stage_builder::Union{Nothing,Function}
    parent::Int
    children::Vector{Int}
    stage::Int
    ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}
    set::Union{AbstractAmbiguitySet, Nothing}
    cost::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}

    function DR_TreeNode(id::Int, parent::Int, stage::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, 
            set::Union{AbstractAmbiguitySet, Nothing}, cost::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}})
        tn = new()
        tn.id = id
        tn.stage_builder = nothing
        tn.parent = parent
        tn.children = Vector{Tuple{Int, Float64}}()
        tn.stage = stage
        tn.ξ = ξ
        tn.set = set
        tn.cost = cost
        return tn
    end
end

function DR_TreeNode(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet, cost::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}})
    return DR_TreeNode(1, 0, 1, ξ, set, cost)
end

get_id(node::DR_TreeNode) = node.id
get_parent(node::DR_TreeNode) = node.parent
get_children(node::DR_TreeNode) = node.children
get_stage(node::DR_TreeNode) = node.stage
get_scenario(node::DR_TreeNode) = node.ξ
get_set(node::DR_TreeNode) = node.set
get_cost(node::DR_TreeNode) = node.cost
function set_stage_builder!(node::DR_TreeNode, func::Function)
    node.stage_builder = func
end

"""
    DR_Tree

Tree keeps information of tree nodes.

    - `nodes`: list of nodes
"""

mutable struct DR_Tree <: DD.AbstractTree
    nodes::Dict{Int,DR_TreeNode}
end

DR_Tree() = DR_Tree(Dict{Int,DR_TreeNode}())

function DR_Tree(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, 
    set::AbstractAmbiguitySet, 
    cost::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}})
    return DR_Tree(Dict{Int,DR_TreeNode}(1 => DR_TreeNode(ξ, set, cost)))
end

function add_child!(tree::DR_Tree, pt::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet, cost::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}})::Int
    #   adds child node to tree.nodes[pt]
    @assert haskey(tree.nodes, pt)                                  # check if pt is valid
    stage = get_stage(tree, pt) + 1                                 # get new stage value
    id = length(tree.nodes) + 1                                     # get node id
    add_node!(tree, DR_TreeNode(id, pt, stage, ξ, set, cost ))      # create node and add to tree
    push!(get_children(tree, pt), id)                               # push id to parent node children
    return id
end

get_children(tree::DR_Tree, id) = get_children(tree.nodes[id])
get_parent(tree::DR_Tree, id) = get_parent(tree.nodes[id])
get_stage(tree::DR_Tree, id) = get_stage(tree.nodes[id])
get_scenario(tree::DR_Tree, id) = get_scenario(tree.nodes[id])
get_set(tree::DR_Tree, id) = get_set(tree.nodes[id])
get_cost(tree::DR_Tree, id) = get_cost(tree.nodes[id])
function set_stage_builder!(tree, id, func::Function)
    set_stage_builder!(tree.nodes[id], func)
end