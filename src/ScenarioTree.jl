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
    coupling_variables::Dict{Any,Vector{DD.CouplingVariableRef}}
    parent::Int
    children::Vector{Int}
    stage::Int
    ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}

    set::Union{AbstractAmbiguitySet, Nothing}
    cost::Dict{String, Float64}

    function DR_TreeNode(id::Int, parent::Int, stage::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, 
            set::Union{AbstractAmbiguitySet, Nothing})
        tn = new()
        tn.id = id
        tn.stage_builder = nothing
        tn.coupling_variables = Dict{Any,Vector{DD.CouplingVariableRef}}()
        tn.parent = parent
        tn.children = Vector{Tuple{Int, Float64}}()
        tn.stage = stage
        tn.ξ = ξ
        tn.set = set
        tn.cost = Dict{Any, Float64}()
        return tn
    end
end

function DR_TreeNode(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet)
    return DR_TreeNode(1, 0, 1, ξ, set)
end

get_set(node::DR_TreeNode) = node.set
get_cost(node::DR_TreeNode, var_id::Any) = node.cost[var_id]
function set_cost!(node::DR_TreeNode, var_id::String, coeff::Float64)
    node.cost[var_id] = coeff
end



function DD.Tree(ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::AbstractAmbiguitySet)
    return DD.Tree(Dict{Int,DR_TreeNode}(1 => DR_TreeNode(ξ, set)))
end

function DD.add_child!(tree::DD.Tree{DR_TreeNode}, pt::Int, ξ::Dict{Symbol, Union{Float64,<:AbstractArray{Float64}}}, set::Union{AbstractAmbiguitySet, Nothing})::Int
    #   adds child node to tree.nodes[pt]
    @assert haskey(tree.nodes, pt)                                  # check if pt is valid
    stage = DD.get_stage(tree, pt) + 1                                 # get new stage value
    id = length(tree.nodes) + 1                                     # get node id
    DD.add_node!(tree, DR_TreeNode(id, pt, stage, ξ, set ))            # create node and add to tree
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
    subtree = DD.SubTree(0)
    # add nodes to subtree
    NDict = Dict{Int, Int}()
    for (id, node) in tree.nodes
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
        subnode.treenode.stage_builder(subtree.model, subnode)
        DD.unregister_all!(subtree.model)

        node = subnode.treenode
        if DD.check_root(subnode)
            next_set = get_set(node)
            obj += subnode.obj + next_set.ϵ * lα__[id] + sum(next_set.samples[s].p * lβ__[id,s] for s = 1:next_set.N )
        elseif DD.check_leaf(subnode)
            this_set = get_set(tree, DD.get_parent(node))
            @constraint(subtree.model, [s=1:this_set.N], this_set.norm_func(this_set.samples[s].ξ, node.ξ) * lα__[DD.get_parent(node)] + lβ__[DD.get_parent(node), s] >= subnode.obj)
        else
            this_set = get_set(tree, DD.get_parent(node))
            next_set = get_set(node)
            @constraint(subtree.model, [s=1:this_set.N], this_set.norm_func(this_set.samples[s].ξ, node.ξ) * lα__[DD.get_parent(node)] + lβ__[DD.get_parent(node), s] >= 
                                                            subnode.obj + next_set.ϵ * lα__[id] + sum(next_set.samples[s].p * lβ__[id,s] for s = 1:next_set.N) )
        end
    end
    JuMP.set_objective(subtree.model, MOI.MIN_SENSE, obj)

    # 
    for (id, subnode) in subtree.nodes
        parent = DD.get_parent(subnode)
        if parent!=0 && haskey(subtree.nodes, parent)
            DD.add_links!(subtree, id, parent)
        end
    end

    subtree.nodelabels = sort!(collect(keys(subtree.nodes)))
    return subtree
end

function DD.create_subtree!(block_id::Int, coupling_variables::Vector{DD.CouplingVariableRef}, nodes::Vector{DR_TreeNode})::DD.SubTree
    subtree = DD.SubTree(block_id)
    # add nodes to subtree
    for node in nodes
        subnode = DD.SubTreeNode(node, 1.0) # dummy weight
        DD.add_node!(subtree, subnode)
        label = DD.get_id(node)
        set_cost!(node, "n$(label)_" * "cobj", 1.0)
    end
    obj = 0
    for (id, subnode) in subtree.nodes
        subnode.treenode.stage_builder(subtree.model, subnode)
        @variable(subtree.model, cobj, DD.ControlInfo, subnode = subnode, ref_symbol = :cobj)
        @constraint(subtree.model, cobj == subnode.obj)
        obj += cobj
        DD.unregister_all!(subtree.model)
    end
    JuMP.set_objective(subtree.model, MOI.MIN_SENSE, obj)

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
        couple_objective!(coupling_variables, block_id, subnode)
    end

    subtree.nodelabels = sort!(collect(keys(subtree.nodes)))
    return subtree
end

function couple_objective!(coupling_variables::Vector{DD.CouplingVariableRef}, block_id::Int, subnode::DD.SubTreeNode)
    label = DD.get_id(subnode)
    DD.couple_variables!(coupling_variables, block_id, label, "cobj", subnode.control["cobj"])
end

"""
    decomposition_not

outputs the entire tree

# Arguments
    - `tree`: Tree
"""


function DD.decomposition_not(tree::DD.Tree{DR_TreeNode}):: Vector{Vector{DR_TreeNode}}
    nodes = Vector{DR_TreeNode}()
    for (id, node) in tree.nodes
        push!(nodes, node)
    end
    return [nodes]
end

"""
    decomposition_scenario

outputs the scenario decomposition at each leaf nodes

# Arguments
    - `tree`: Tree
"""

function DD.decomposition_scenario(tree::DD.Tree{DR_TreeNode}):: Vector{Vector{DR_TreeNode}}
    node_cluster = Vector{Vector{DR_TreeNode}}()
    for (id, node) in tree.nodes
        if DD.check_leaf(node)
            nodes = Vector{DR_TreeNode}()
            current = node
            while true
                push!(nodes, current)
                pt = DD.get_parent(current)
                current = tree.nodes[pt]
                if DD.check_root(current)
                    push!(nodes, current)
                    break
                end
            end
            push!(node_cluster, nodes)
        end
    end
    return node_cluster
end

"""
    decomposition_temporal

outputs the temporal decomposition at each nodes

# Arguments
    - `tree`: Tree
"""
#need fix

function DD.decomposition_temporal(tree::DD.Tree{DR_TreeNode}):: Vector{Vector{DR_TreeNode}}
    node_cluster = Vector{Vector{TreeNode}}()
    for (id, node) in tree.nodes
        push!(node_cluster,[node])
    end
    return node_cluster
end