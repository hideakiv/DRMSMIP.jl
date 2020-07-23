using JuMP, Ipopt, Gurobi
using DRMSMIP
using DualDecomposition
using Random

const DD = DualDecomposition

const rng = Random.MersenneTwister(1234)
"""
a: interest rate
π: unit stock price
ρ: unit dividend price


K = 3 number of stages
L = 2 number of investment vehicles
2^L scenarios in each stage
2^L^(K-1)=16 scenarios in total
ρ = 0.05 * π
bank: interest rate 0.01
asset1: 1.03 or 0.97
asset2: 1.06 or 0.94

In each node, we have Np=10 samples from a log-normal distribution
"""

function create_tree(K::Int, L::Int, Np::Int)::DRMSMIP.Tree
    π = ones(L)
    π_samp = generate_sample(L, π, Np)
    set = DRMSMIP.WassersteinSet(π_samp, 1.0, norm_L1)
    cost = zeros(L+1)
    tree = DRMSMIP.Tree(π, set, cost)
    add_nodes!(K, L, tree, 1, 1, Np)
    return tree
end


function generate_sample(L::Int, π::Array{Float64}, Np::Int)::Array{DRMSMIP.Sample}
    # generates random samples following a lognormal distribution
    ret = Array{DRMSMIP.Sample}(undef, Np)
    for ii in 1:Np
        ξ = Array{Float64}(undef, L)
        for l in 1:L
            sig = sqrt( log( 0.5+sqrt( ( 0.03*l )^2+0.25 ) ) )
            rnd = sig * randn(rng) .+ log(π[l])
            ξ[l] = exp(rnd)
        end
        ret[ii] = DRMSMIP.Sample(ξ, 1/Np)
    end
    return ret
end

function add_nodes!(K::Int, L::Int, tree::DRMSMIP.Tree, id::Int, k::Int, Np::Int)
    if k < K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            π_samp = generate_sample(L, π, Np)
            set = DRMSMIP.WassersteinSet(π_samp, 1.0, norm_L1)
            cost = zeros(L+1)
            DRMSMIP.addchild!(tree, id, π, set, cost)
            childid = length(tree.nodes)
            add_nodes!(K, L, tree, childid, k+1, Np)
        end
    elseif k == K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            cost = vcat(-π, [-1])
            DRMSMIP.addchild!(tree, id, π, nothing, cost)
        end
    end
end

function iterlist(L::Int, π::Array{Float64})::Array{Array{Float64}}
    # generates all combinations of up and down scenarios
    ls = [Float64[] for _ in 1:2^L]
    ii = 1

    function foo(L::Int, l::Int, arr::Vector{Float64})
        up = (1.0 + 0.03 * l) * π[l]
        dn = (1.0 - 0.03 * l) * π[l]

        if l < L
            arr1 = copy(arr)
            arr1[l] = up
            foo(L, l+1, arr1)

            arr2 = copy(arr)
            arr2[l] = dn
            foo(L, l+1, arr2)
        else
            arr1 = copy(arr)
            arr1[l] = up
            ls[ii] = arr1
            ii+=1

            arr2 = copy(arr)
            arr2[l] = dn
            ls[ii] = arr2
            ii+=1
        end
    end

    foo(L, 1, Array{Float64}(undef, L))
    return ls
end

"""
    max     B_K+∑_{l=1}^{L}π_{K,l}y_{K,l}

    s.t.    B_1+∑_{l=1}^{L}π_{1,l}x_{1,l} = b_1

            b_k+(1+a)B_{k-1}+∑_{l=1}^{L}ρ_{k,l}y_{k-1,l} = B_k+∑_{l=1}^{L}π_{k,l}x_{k,l}, ∀ k=2,…,K
    
            y_{1,l} = x_{1,l}, ∀ l=1,…,L
    
            y_{k-1,l}+x_{k,l} = y_{k,l}, ∀ k=2,…,K, l=1,…,L
    
            x_{k,l} ∈ ℤ , ∀ k=1,…,K, l=1,…,L
    
            y_{k,l} ≥ 0, ∀ k=1,…,K, l=1,…,L
    
            B_k ≥ 0, ∀ k=1,…,K.
"""
const K = 3
const L = 2
const a = 0.01
const b = [100, 30, 30]
const Np = 10

function create_scenario_model(K::Int, L::Int, tree::DRMSMIP.Tree, id::Int)
    hist = DRMSMIP.get_history(tree, id)
    m = Model(Gurobi.Optimizer) 
    set_optimizer_attribute(m, "OutputFlag", 0)
    @variable(m, x[1:K,1:L], integer=true)
    @variable(m, y[1:K,1:L]>=0)
    @variable(m, B[1:K]>=0)

    π = tree.nodes[1].ξ

    con = @constraint(m, B[1] + sum( π[l] * x[1,l] for l in 1:L) == b[1])
    set_name(con, "con[1]")

    for l in 1:L
        bal = @constraint(m, y[1,l]-x[1,l]==0)
        set_name(bal, "bal[1,$(l)]")
    end

    for k = 2:K
        π = tree.nodes[hist[k]].ξ
        ρ = tree.nodes[hist[k-1]].ξ * 0.05

        con = @constraint(m, B[k] + sum( π[l] * x[k,l] - ρ[l] * y[k,l] for l in 1:L)
            - (1+a) * B[k-1] == b[k])
        set_name(con, "con[$(k)]")
        for l in 1:L
            bal = @constraint(m, y[k,l]-x[k,l]-y[k-1,l]==0)
            set_name(bal, "bal[$(k),$(l)]")
        end
    end
    #π = tree.nodes[id].ξ
    #@objective(m, Max, B[K] + sum( π[l] * x[K,l] for l in 1:L )
    @objective(m, Min, 0 )
    return m
end

function leaf2block(nodes::Array{Int})::Dict{Int,Int}
    leafdict = Dict{Int,Int}()
    for i in 1:length(nodes)
        id = nodes[i]
        leafdict[id] = i
    end
    return leafdict
end


function norm_L1(x::Array{Float64}, y::Array{Float64})::Float64
    val = 0
    for i in 1:length(x)
        val += abs(x[i] - y[i])
    end
    return val
end


function main_dd()
    tree = create_tree(K,L,Np)

    # Create DualDecomposition instance.
    algo = DRMSMIP.DRMS_LagrangeDual(tree)

    # Add Lagrange dual problem for each scenario s.
    nodes = DRMSMIP.get_stage_id(tree)
    leafdict = leaf2block(nodes[K])
    models = Dict{Int,JuMP.Model}(id => create_scenario_model(K,L,tree,id) for id in nodes[K])
    for id in nodes[K]
        DD.add_block_model!(algo, leafdict[id], models[id])
    end

    coupling_variables = Vector{DD.CouplingVariableRef}()
    for k in 1:K-1
        for root in nodes[k]
            leaves = DRMSMIP.get_future(tree, root)
            for id in leaves
                model = models[id]
                xref = model[:x]
                for l in 1:L
                    push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, l], xref[k, l]))
                end
                Bref = model[:B]
                push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, L+1], Bref[k]))
            end
        end
    end
    # dummy coupling variables
    for id in nodes[K]
        model = models[id]
        xref = model[:x]
        for l in 1:L
            push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [id, l], xref[K, l]))
        end
        Bref = model[:B]
        push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [id, L+1], Bref[K]))
    end

    # Set nonanticipativity variables as an array of symbols.
    DD.set_coupling_variables!(algo, coupling_variables)


    # Solve the problem with the solver; this solver is for the underlying bundle method.
    #DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))


    DeterministicEq(L, tree)
end


function DeterministicEq(L::Int, tree::DRMSMIP.Tree)
    m = Model(Gurobi.Optimizer) 
    lenN = length(tree.nodes)
    node = tree.nodes[1]
    @variable(m, x[1:lenN,1:L], integer=true)
    @variable(m, y[1:lenN,1:L]>=0)
    @variable(m, B[1:lenN]>=0)
    @variable(m, α[1:lenN]>=0)
    @variable(m, β[1:lenN,1:node.set.N])

    @objective(m, Min, sum( node.cost[l]*x[1,l] for l in 1:L) + node.cost[L+1]*B[1]
         + node.set.ϵ*α[1] + sum( node.set.samples[ss].p*β[1,ss] for ss in 1:node.set.N) )

    π = tree.nodes[1].ξ
    @constraint(m, B[1] + sum( π[l] * x[1,l] for l in 1:L) == b[1])
    for l in 1:L
        @constraint(m, y[1,l]-x[1,l]==0)
    end

    function iterate_children(id::Int)
        pnode = tree.nodes[id]
        children = DRMSMIP.get_children(tree, id)
        
        for child in children
            cnode = tree.nodes[child]

            π = cnode.ξ
            ρ = pnode.ξ * 0.05

            @constraint(m, B[child] + sum( π[l] * x[child,l] - ρ[l] * y[child,l] for l in 1:L)
                - (1+a) * B[id] == b[cnode.k])
            for l in 1:L
                @constraint(m, y[child,l]-x[child,l]-y[id,l]==0)
            end

            if length(DRMSMIP.get_children(tree, child)) > 0
                for s in 1:pnode.set.N
                    foo = pnode.set.norm_func(cnode.ξ, pnode.set.samples[s].ξ)
                    @constraint(m, foo*α[id] + β[id,s]
                        - sum( cnode.cost[l]*x[child,l] for l in 1:L) - cnode.cost[L+1]*B[child] - cnode.set.ϵ*α[child]
                        - sum( cnode.set.samples[ss].p*β[child,ss] for ss in 1:cnode.set.N) >= 0)
                end
            else
                for s in 1:pnode.set.N
                    foo = pnode.set.norm_func(cnode.ξ, pnode.set.samples[s].ξ)
                    @constraint(m, foo*α[id] + β[id,s]
                        - sum( cnode.cost[l]*x[child,l] for l in 1:L) - cnode.cost[L+1]*B[child] >= 0)
                end
            end
            iterate_children(child)
        end
        
    end
    iterate_children(1)
    JuMP.optimize!(m)
    open("model.lp", "w") do f
        print(f, m)
    end
    println(value.(m[:x]))
    println(value.(m[:B]))
    println(value.(m[:y]))
end