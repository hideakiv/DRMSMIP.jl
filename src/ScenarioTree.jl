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
    cost::Dict{Any, Float64}

    function DR_TreeNode(id::Int, parent::Int, stage::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, 
            set::Union{AbstractAmbiguitySet, Nothing})
        tn = new()
        tn.id = id
        tn.stage_builder = nothing
        tn.parent = parent
        tn.children = Vector{Tuple{Int, Float64}}()
        tn.stage = stage
        tn.ξ = ξ
        tn.set = set
        tn.cost = Dict{Ant, Float64}()
        return tn
    end
end

function DR_TreeNode(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet)
    return DR_TreeNode(1, 0, 1, ξ, set)
end

get_set(node::DR_TreeNode) = node.set
get_cost(node::DR_TreeNode, var_id::Any) = node.cost[var_id]
function set_cost!(node::DR_TreeNode, var_id::Any, coeff::Float64)
    node.cost[var_id] = coeff
end


function DD.Tree(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet)
    return DD.Tree(Dict{Int,DR_TreeNode}(1 => DR_TreeNode(ξ, set)))
end

function add_child!(tree::DD.Tree{DR_TreeNode}, pt::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet)::Int
    #   adds child node to tree.nodes[pt]
    @assert haskey(tree.nodes, pt)                                  # check if pt is valid
    stage = DD.get_stage(tree, pt) + 1                                 # get new stage value
    id = length(tree.nodes) + 1                                     # get node id
    add_node!(tree, DR_TreeNode(id, pt, stage, ξ, set ))            # create node and add to tree
    push!(DD.get_children(tree, pt), id)                               # push id to parent node children
    return id
end

get_set(tree::DD.Tree{DR_TreeNode}, id) = get_set(tree.nodes[id])

"""
    create_Wasserstein_deterministic!

creates deterministic model

# Arguments
    - `tree`: Tree{DR_TreeNode}
"""

function create_Wasserstein_deterministic!(tree::DD.Tree{DR_TreeNode})
    subtree = SubTree(0)
    # add nodes to subtree
    NDict = Dict{Int, Int}()
    for (id, node) in tree.nodes
        @assert typeof(node.set) == WassersteinSet
        subnode = DD.SubTreeNode(node, 0.0)
        DD.add_node!(subtree, subnode)
        if !DD.check_leaf(node)
            NDict[id] = node.set.N
        end
    end

    @variable(subtree.model, lα__[id = keys(NDict)] >= 0)
    @variable(subtree.model, lβ__[id = keys(NDict), s=1:NDict[id]])

    obj = 0
    for (id, subnode) in subtree.nodes
        subnode.treenode.stage_builder(tree, subtree, subnode) # make macro for changing variable names and constraint names to include node id

        node = subnode.treenode
        cost = sum(coeff * var for (var, coeff) in subnode.cost)
        if DD.check_root(subnode)
            next_set = get_set(node)
            obj += cost + next_set.ϵ * lα[id] + sum(next_set.samples[s].p * lβ[id,s] for s = 1:next_set.N )
        elseif DD.check_leaf(subnode)
            this_set = get_set(tree, DD.get_parent(node))
            @constraint(subtree.model, [s=1:this_set.N], this_set.norm_func(this_set.ξ, node.ξ) * lα__[DD.get_parent(node)] + lβ__[DD.get_parent(node), s] >= cost)
        else
            this_set = get_set(tree, DD.get_parent(node))
            next_set = get_set(node)
            @constraint(subtree.model, [s=1:this_set.N], this_set.norm_func(this_set.ξ, node.ξ) * lα__[DD.get_parent(node)] + lβ__[DD.get_parent(node), s] >= 
                                                            cost + next_set.ϵ * lα[id] + sum(next_set.samples[s].p * lβ[id,s] for s = 1:next_set.N) )
        end
    end
    set_objective!(subtree, MOI.MIN_SENSE, obj)

    # 
    for (id, subnode) in subtree.nodes
        parent = DD.get_parent(subnode)
        if parent!=0 && haskey(subtree.nodes, parent)
            DD.add_links!(subtree, id, parent)
        end
    end
    return subtree
end

function create_subtree!(tree::DD.Tree{DR_TreeNode}, block_id::Int, coupling_variables::Vector{DD.CouplingVariableRef}, nodes::Vector{Tuple{DR_TreeNode,Float64}})::DD.SubTree
    subtree = DD.SubTree(block_id)
    # add nodes to subtree
    for (node, weight) in nodes
        subnode = DD.SubTreeNode(node, weight)
        DD.add_node!(subtree, subnode)
    end

    for (id, subnode) in subtree.nodes
        subnode.treenode.stage_builder(tree, subtree, subnode) # make macro for changing variable names and constraint names to include node id
        for (symb, var) in subnode.control
            set_objective_coefficient(subtree.model, subnode, var)
        end
    end
    JuMP.set_objective_sense(subtree.model, MOI.MIN_SENSE)

    # 
    for (id, subnode) in subtree.nodes
        DD.couple_common_variables!(coupling_variables, block_id, subnode)
        parent = DD.get_parent(subnode)
        if parent!=0 && haskey(subtree.nodes, parent)
            DD.add_links!(subtree, id, parent)
        elseif parent!=0 # assuming 1st stage node is 1
            subtree.parent = parent
            subtree.root = id
            DD.couple_incoming_variables!(coupling_variables, block_id, subnode)
        end
    end
    return subtree
end

function set_objective_coefficient(model::JuMP.Model, subnode::DD.SubTreeNode, var::JuMP.VariableRef)
    if haskey(subnode.cost, var)
        coeff = subnode.cost[var]
        JuMP.set_objective_coefficient(model, var, subnode.weight * coeff)
    end
end

function set_objective_coefficient(model::JuMP.Model, subnode::DD.SubTreeNode, vars::AbstractArray{JuMP.VariableRef})
    for key in eachindex(vars)
        set_objective_coefficient(model, subnode, vars[key])
    end
end