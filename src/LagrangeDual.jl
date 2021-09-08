"""
DR_LagrangeDual

Lagrangian dual method for dual decomposition. This `mutable struct` constains:
- `block_model::BlockModel` object
- `var_to_index` mapping coupling variable to the index wrt the master problem
- `masiter::Int` sets the maximum number of iterations
- `tol::Float64` sets the relative tolerance for termination
- `tree` keeps Tree information
"""

mutable struct DR_LagrangeDual <: DD.AbstractLagrangeDual
    block_model::DR_BlockModel
    var_to_index::Dict{Tuple{Int,Any},Int} # maps coupling variable to the index wrt the master problem
    heuristics::Vector{Type}
    subsolve_time::Vector{Dict{Int,Float64}}
    subcomm_time::Vector{Float64}
    subobj_value::Vector{Float64}
    master_time::Vector{Float64}

    tree::DD.Tree{DR_TreeNode} # TODO: abstraction to non-tree problem

    function DR_LagrangeDual(tree::DD.Tree{DR_TreeNode})
        LD = new()
        LD.block_model = DR_BlockModel()
        LD.var_to_index = Dict()
        LD.heuristics = []
        LD.subsolve_time = []
        LD.subcomm_time = []
        LD.subobj_value = []
        LD.master_time = []

        LD.tree = tree #
        
        return LD
    end
end

function DD.get_solution!(LD::DR_LagrangeDual, method::BM.AbstractMethod)
    LD.block_model.dual_solution = copy(BM.get_solution(method))
    bundle = BM.get_model(method)
    model = BM.get_model(bundle)
    P = model[:P]
    for id in axes(P)[1]
        LD.block_model.P_solution[id] = JuMP.value(P[id])
    end
end

function DD.add_constraints!(LD::DR_LagrangeDual, method::DD.BundleMaster)
    node_to_couple = sort_couple_by_label(LD.tree, LD.block_model.variables_by_couple)

    model = BM.get_jump_model(method)
    @variable(model, P[2:length(LD.tree.nodes)] >= 0)

    for (id, node) in LD.tree.nodes
        add_non_anticipativity!(LD, model, node, node_to_couple[id])
        add_ambiguity!(LD.tree, model, node, node.set)
    end
end

function sort_couple_by_label(tree::DD.Tree, variables_by_couple::Dict{Any,Vector{DD.CouplingVariableKey}})::Dict{Int,Vector{Any}}
    node_to_couple = Dict{Int, Vector{Any}}()
    for (id, nodes) in tree.nodes
        node_to_couple[id] = Vector{Any}()
    end
    for (couple_id, keys) in variables_by_couple
        node_id = couple_id[1]
        push!(node_to_couple[node_id], couple_id)
    end
    return node_to_couple
end

function add_non_anticipativity!(LD::DR_LagrangeDual, m::JuMP.Model, node::DR_TreeNode, couple_ids::Vector{Any})
    λ = m[:x]
    P = m[:P]

    if DD.check_root(node)
        for couple_id in couple_ids
            vars = LD.block_model.variables_by_couple[couple_id]
            @constraint(m, sum(λ[DD.index_of_λ(LD, v)] for v in vars) == get_cost(node, couple_id))
        end
    else
        for couple_id in couple_ids
            vars = LD.block_model.variables_by_couple[couple_id]
            @constraint(m, sum(λ[DD.index_of_λ(LD, v)] for v in vars) == get_cost(node, couple_id) * P[DD.get_id(node)])
        end
    end
end


function add_ambiguity!(tree::DD.Tree{DR_TreeNode}, m::JuMP.Model, node::DR_TreeNode, set::AbstractAmbiguitySet) end
function add_ambiguity!(tree::DD.Tree{DR_TreeNode}, m::JuMP.Model, node::DR_TreeNode, set::Nothing) end

function add_ambiguity!(tree::DD.Tree{DR_TreeNode}, m::JuMP.Model, node::DR_TreeNode, set::WassersteinSet)
    P = m[:P]
    if DD.check_root(node)
        @variable(model, w[id=DD.get_children(node),s=1:set.N] >= 0, base_name = "n1_w")
        @constraint(m, sum( sum( w[id, s] * set.norm_func(tree.nodes[id].ξ, set.samples[s].ξ) for s in 1:set.N) for id in DD.get_children(node)) <= set.ϵ)
        @constraint(m, [s=1:set.N], sum( w[id, s] for id in DD.get_children(node)) == set.samples[s].p )
        @constraint(m, [id=DD.get_children(node)], sum( w[id, s] for s in 1:set.N) == P[id])
        @constraint(m, sum( P[child] for child in DD.get_children(node)) == 1)
        JuMP.unregister(mdl, :w)
    else #shouldn't arrive at leaf node
        @variable(model, w[id=DD.get_children(node),s=1:set.N] >= 0, base_name = "n$(get_id(node))_w")
        @constraint(m, sum( sum( w[id, s] * set.norm_func(tree.nodes[id].ξ, set.samples[s].ξ) for s in 1:set.N) for id in DD.get_children(node)) <= set.ϵ * P[DD.get_id(node)])
        @constraint(m, [s=1:set.N], sum( w[id, s] for id in DD.get_children(node)) == set.samples[s].p * P[DD.get_id(node)])
        @constraint(m, [id=DD.get_children(node)], sum( w[id, s] for s in 1:set.N) == P[id])
        @constraint(m, sum( P[child] for child in DD.get_children(node)) == P[DD.get_id(node)])
        JuMP.unregister(mdl, :w)
    end
end

function initialize_bundle(tree::DD.Tree{DR_TreeNode}, LD::DR_LagrangeDual)::Array{Float64,1}
    n = DD.parallel.sum(DD.num_coupling_variables(LD.block_model))
    bundle_init = Array{Float64,1}(undef, n)
    variable_keys = [v.key for v in LD.block_model.coupling_variables]
    all_variable_keys = DD.parallel.allcollect(variable_keys)
    if DD.parallel.is_root()
        P = get_feasible_P(tree)
        for key in all_variable_keys
            i = LD.var_to_index[(key.block_id,key.coupling_id)]
            N = length(LD.block_model.variables_by_couple[key.coupling_id])
            block_id = key.block_id
            node_id = key.coupling_id[1]
            couple_id = key.coupling_id
            if DD.check_root(tree.nodes[node_id])
                bundle_init[i] =  tree.nodes[node_id].cost[couple_id] / N
            else
                bundle_init[i] =  tree.nodes[node_id].cost[couple_id] / N * P[node_id] #P[corresponding leaf node] ?
            end
        end
        DD.parallel.bcast(bundle_init)
    else
        bundle_init = DD.parallel.bcast(nothing)
    end
    return bundle_init
end

function get_feasible_P(tree::DD.Tree{DR_TreeNode})::Dict{Int,Float64}
    model = JuMP.Model(GLPK.Optimizer)

    @variable(model, P[2:length(tree.nodes)] >= 0)
    for (id, node) in tree.nodes
        add_ambiguity!(tree, model, node, node.set)
    end

    @objective(model, Min, 0)

    JuMP.optimize!(model)

    Pref = Dict()
    P = model[:P]

    for id in 2:length(tree.nodes)
        Pref[id] = JuMP.value(P[id])
    end
    return Pref
end