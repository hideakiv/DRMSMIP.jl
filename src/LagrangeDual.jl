"""
DRMS_LagrangeDual

Lagrangian dual method for dual decomposition. This `mutable struct` constains:
- `block_model::BlockModel` object
- `var_to_index` mapping coupling variable to the index wrt the master problem
- `masiter::Int` sets the maximum number of iterations
- `tol::Float64` sets the relative tolerance for termination
- `tree` keeps Tree information
"""

mutable struct DRMS_LagrangeDual{T<:BM.AbstractMethod} <: DD.AbstractLagrangeDual
    block_model::DRMS_BlockModel
    var_to_index::Dict{Tuple{Int,Any},Int} # maps coupling variable to the index wrt the master problem
    bundle_method
    maxiter::Int # maximum number of iterations
    tol::Float64 # convergence tolerance

    tree::Tree

    function DRMS_LagrangeDual(tree::Tree, T = BM.ProximalMethod, 
            maxiter::Int = 1000, tol::Float64 = 1e-6)
        LD = new{T}()
        LD.block_model = DRMS_BlockModel()
        LD.var_to_index = Dict()
        LD.bundle_method = T
        LD.maxiter = maxiter
        LD.tol = tol

        LD.tree = tree

        DD.parallel.init()
        finalizer(DD.finalize!, LD)
        
        return LD
    end
end

function DD.get_solution!(LD::DRMS_LagrangeDual, method::BM.AbstractMethod)
    LD.block_model.dual_solution = copy(BM.get_solution(method))
    bundle = BM.get_model(method)
    model = BM.get_model(bundle)
    P = model[:P]
    for id in 2:length(LD.tree.nodes)
        LD.block_model.P_solution[id] = JuMP.value(P[id])
    end
end

function DD.add_constraints!(LD::DRMS_LagrangeDual, method::BM.AbstractMethod)
    model = BM.get_jump_model(method)

    add_Wasserstein!(LD, model)
end

function add_Wasserstein!(LD::DRMS_LagrangeDual, model::JuMP.Model)
    K = LD.tree.K
    nodelist = get_stage_id(LD.tree)

    @variable(model, P[2:length(LD.tree.nodes)] >= 0)
    @variable(model, w[k=1:K-1, id=nodelist[k+1], s=1:LD.tree.nodes[get_parent(LD.tree,id)].set.N] >= 0)
    
    con_X!(LD.tree, model, nodelist, K, LD)
    con_E!(LD.tree, model, nodelist, K)
    con_P!(LD.tree, model, nodelist, K)
    con_M!(LD.tree, model, nodelist, K)
    con_N!(LD.tree, model, nodelist, K)
end

function con_X!(tree::Tree, m::Model, nodelist::Array{Array{Int}}, K::Int, LD::DRMS_LagrangeDual)
    #   Lagrangian dual of nonanticipativity of x
    λ = m[:x]
    P = m[:P]

    c = tree.nodes[1].cost
    Nx = length(c)

    for ix in 1:Nx
        vars = LD.block_model.variables_by_couple[[1, ix]]
        con_x = @constraint(m, sum(λ[DD.index_of_λ(LD, v)] for v in vars) == c[ix])
        set_name(con_x, "con_x[1,1,$(ix)]")
    end

    for k = 2:K
        for root in nodelist[k]
            c = tree.nodes[root].cost
            Nx = length(c)
            for ix in 1:Nx
                vars = LD.block_model.variables_by_couple[[root, ix]]
                con_x = @constraint(m,
                    sum(λ[DD.index_of_λ(LD, v)] for v in vars) - c[ix] * P[root] == 0)
                set_name(con_x, "con_x[$(k),$(root),$(ix)]")
            end
        end
    end
end

function con_E!(tree::Tree, m::Model, nodelist::Array{Array{Int}}, K::Int)
    #   constraints involving e
    w = m[:w]
    P = m[:P]

    rootnode = tree.nodes[1]
    Np = rootnode.set.N
    con_e = @constraint(m,
        sum( sum( w[1, id, s] * rootnode.set.norm_func(tree.nodes[id].ξ, rootnode.set.samples[s].ξ) for s in 1:Np) for id in rootnode.children) 
        <= rootnode.set.ϵ)
    set_name(con_e, "con_e[1,1]")

    for k = 2:K-1
        for root in nodelist[k]
            rootnode = tree.nodes[root]
            Np = rootnode.set.N
            con_e = @constraint(m,
                sum( sum( w[k, id, s] * rootnode.set.norm_func(tree.nodes[id].ξ, rootnode.set.samples[s].ξ) for s in 1:Np) for id in rootnode.children) 
                - rootnode.set.ϵ * P[root] <= 0)
            set_name(con_e, "con_e[$(k),$(root)]")
        end
    end
end

function con_P!(tree::Tree, m::Model, nodelist::Array{Array{Int}}, K::Int)
    #   constraints involving p
    w = m[:w]
    P = m[:P]

    rootnode = tree.nodes[1]
    Np = rootnode.set.N
    for s in 1:Np
        con_p = @constraint(m,
            sum( w[1, id, s] for id in rootnode.children) == rootnode.set.samples[s].p )
        set_name(con_p, "con_p[1,1,$(s)]")
    end

    for k = 2:K-1
        for root in nodelist[k]
            rootnode = tree.nodes[root]
            Np = rootnode.set.N
            for s in 1:Np
                con_p = @constraint(m,
                    sum( w[k, id, s] for id in rootnode.children) - rootnode.set.samples[s].p * P[root] == 0 )
                set_name(con_p, "con_p[$(k),$(root),$(s)]")
            end
        end
    end
end

function con_M!(tree::Tree, m::Model, nodelist::Array{Array{Int}}, K::Int)
    #   constraints for marginal probability
    w = m[:w]
    P = m[:P]

    for k = 1:K-1
        for root in nodelist[k]
            rootnode = tree.nodes[root]
            for id in rootnode.children
                Np = rootnode.set.N
                con_m = @constraint(m,
                    sum( w[k, id, s] for s in 1:Np) - P[id] == 0)
                set_name(con_m, "con_m[$(k),$(root),$(id)]")
            end
        end
    end
end

function con_N!(tree::Tree, m::Model, nodelist::Array{Array{Int}}, K::Int)
    #   constraints for normalization
    P = m[:P]
    con_n = @constraint(m,
        sum( P[child] for child in get_children(tree, 1)) == 1)
    set_name(con_n, "con_n[1]")
    for k = 2:K-1
        for root in nodelist[k]
            con_n = @constraint(m,
                sum( P[child] for child in get_children(tree, root)) == P[root])
            set_name(con_n, "con_n[$(root)]")
        end
    end
end