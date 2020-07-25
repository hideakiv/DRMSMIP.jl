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

"""

function create_tree(K::Int, L::Int, Np::Int)::DRMSMIP.Tree
    π = ones(L)
    π_samp = generate_sample(L, π, Np)                  #igonre this for stochastic problem
    set = DRMSMIP.WassersteinSet(π_samp, 0.0, norm_L1)  #ignore this for stochastic problem
    cost = zeros(L+1)                                   #ignore this for stochastic problem
    tree = DRMSMIP.Tree(π, set, cost)
    add_nodes!(K, L, tree, 1, 1, Np)
    return tree
end


function generate_sample(L::Int, π::Array{Float64}, Np::Int)::Array{DRMSMIP.Sample}
    # generates random samples following a lognormal distribution
    ret = Array{DRMSMIP.Sample}(undef, 2^L)
    ls = iterlist(L, π)
    for ii in 1:2^L
        ξ = Array{Float64}(undef, L)
        for l in 1:L
            ξ[l] = ls[ii][l]
        end
        ret[ii] = DRMSMIP.Sample(ξ, 1/(2^L))
    end
    return ret
end

function add_nodes!(K::Int, L::Int, tree::DRMSMIP.Tree, id::Int, k::Int, Np::Int)
    if k < K-1
        ls = iterlist(L,tree.nodes[id].ξ)
        for π in ls
            π_samp = generate_sample(L, π, Np)
            set = DRMSMIP.WassersteinSet(π_samp, 0.0, norm_L1)
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
    #@variable(m, x[1:K,1:L], integer=true)
    @variable(m, x[1:K,1:L])
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

        con = @constraint(m, B[k] + sum( π[l] * x[k,l] - ρ[l] * y[k-1,l] for l in 1:L)
            - (1+a) * B[k-1] == b[k])
        set_name(con, "con[$(k)]")
        for l in 1:L
            bal = @constraint(m, y[k,l]-x[k,l]-y[k-1,l]==0)
            set_name(bal, "bal[$(k),$(l)]")
        end
    end
    for l in 1:L
        @constraint(m, x[K,l]==0)
    end
    π = tree.nodes[id].ξ
    @objective(m, Min, - (B[K] + sum( π[l] * y[K,l] for l in 1:L ))/(2^L)^(K-1) )
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


function main_comp()
    tree = create_tree(K,L,Np)

    LD = dual_decomp(L, tree)

    NAmodel = non_anticipative(L,tree)
    non_anticipative_results(tree,NAmodel)
end


function dual_decomp(L::Int, tree::DRMSMIP.Tree)
    # Create DualDecomposition instance.
    algo = DD.LagrangeDual(BM.TrustRegionMethod)

    # Add Lagrange dual problem for each scenario s.
    nodelist = DRMSMIP.get_stage_id(tree)
    leafdict = leaf2block(nodelist[K])
    models = Dict{Int,JuMP.Model}(id => create_scenario_model(K,L,tree,id) for id in nodelist[K])
    for id in nodelist[K]
        DD.add_block_model!(algo, leafdict[id], models[id])
    end

    coupling_variables = Vector{DD.CouplingVariableRef}()
    for k in 1:K-1
        for root in nodelist[k]
            leaves = DRMSMIP.get_future(tree, root)
            for id in leaves
                model = models[id]
                yref = model[:y]
                for l in 1:L
                    push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, l], yref[k, l]))
                end
                Bref = model[:B]
                push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, L+1], Bref[k]))
            end
        end
    end
    # dummy coupling variables
    for id in nodelist[K]
        model = models[id]
        yref = model[:y]
        for l in 1:L
            push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [id, l], yref[K, l]))
        end
        Bref = model[:B]
        push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [id, L+1], Bref[K]))
    end

    # Set nonanticipativity variables as an array of symbols.
    DD.set_coupling_variables!(algo, coupling_variables)

    # Solve the problem with the solver; this solver is for the underlying bundle method.
    DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    return algo
end

function interpret_solution(tree::DRMSMIP.Tree, LD::DRMSMIP.DRMS_LagrangeDual, sol::Array{Float64,1})
    n = DD.num_coupling_variables(LD.block_model)
    #sol = LD.block_model.dual_solution
    nodelist = DRMSMIP.get_stage_id(tree)
    solret = Dict()
    for i in 1:n
        key = LD.block_model.coupling_variables[i].key
        sce = key.block_id
        root_id = key.coupling_id[1]
        l = key.coupling_id[2]
        k = tree.nodes[root_id].k
        solret[k,nodelist[K][sce],l] = sol[i]
    end
    return solret
end

function convert_lp(tree::DRMSMIP.Tree, LD::DRMSMIP.DRMS_LagrangeDual)
    input = "examples/investment_stochastic_results_v2/dual_decomp.lp"
    lines = readlines(input)
    output = "examples/investment_stochastic_results_v2/dual_decomp_v2.lp"
    nodelist = DRMSMIP.get_stage_id(tree)
    open(output, "w") do io
        for line in lines
            foo = split(line)
            for i in 1:length(foo)
                elem = foo[i]
                if elem[1] == 'x'
                    num = parse(Int,elem[3:end-1])
                    key = LD.block_model.coupling_variables[num].key
                    sce = key.block_id
                    root_id = key.coupling_id[1]
                    l = key.coupling_id[2]
                    k = tree.nodes[root_id].k
                    foo[i] = "x[$(k),$(nodelist[K][sce]),$(l)]"
                end
            end
            ret = join(foo, " ")
            write(io, ret*"\n")
        end
    end
end




function non_anticipative(L::Int, tree::DRMSMIP.Tree)
    nodelist = DRMSMIP.get_stage_id(tree)
    m = Model(Gurobi.Optimizer) 
    set_optimizer_attribute(m, "OutputFlag", 0)

    #@variable(m, x[1:K,1:L], integer=true)
    @variable(m, x[nodelist[K],1:K,1:L])
    @variable(m, y[nodelist[K],1:K,1:L]>=0)
    @variable(m, B[nodelist[K],1:K]>=0)

    @variable(m, yp[k=1:K,nodelist[k],1:L]>=0)
    @variable(m, Bp[k=1:K,nodelist[k]]>=0)

    for id in nodelist[K]
        hist = DRMSMIP.get_history(tree, id)

        π = tree.nodes[1].ξ

        @constraint(m, B[id,1] + sum( π[l] * x[id,1,l] for l in 1:L) == b[1])

        for l in 1:L
            @constraint(m, y[id,1,l]-x[id,1,l]==0)
        end

        for k = 2:K
            π = tree.nodes[hist[k]].ξ
            ρ = tree.nodes[hist[k-1]].ξ * 0.05

            @constraint(m, B[id,k] + sum( π[l] * x[id,k,l] - ρ[l] * y[id,k-1,l] for l in 1:L)
                - (1+a) * B[id,k-1] == b[k])
            for l in 1:L
                @constraint(m, y[id,k,l]-x[id,k,l]-y[id,k-1,l]==0)
            end
        end
        for l in 1:L
            @constraint(m, x[id,K,l]==0)
        end
        for k = 1:K
            root = hist[k]
            for l in 1:L
                con_na = @constraint(m, y[id,k,l]==yp[k,root,l])
                set_name(con_na, "con_na[$(k),$(id),$(l)]")
            end
            con_na = @constraint(m, B[id,k]==Bp[k,root])
            set_name(con_na, "con_na[$(k),$(id),$(L+1)]")
        end
    end
    @objective(m, Min, - sum( Bp[K,id] + sum( tree.nodes[id].ξ[l] * yp[K,id,l] for l in 1:L)
         for id in nodelist[K] )/ length(nodelist[K]) )
    JuMP.optimize!(m)
    return m
end

function non_anticipative_results(tree::DRMSMIP.Tree, model::Model)
    open("examples/investment_stochastic_results_v2/non_anticipative.lp", "w") do f
        print(f, model)
    end
    nodelist = DRMSMIP.get_stage_id(tree)
    
    xref = value.(model[:x])
    Bref = value.(model[:B])
    yref = value.(model[:y])

    open("examples/investment_stochastic_results_v2/non_anticipative_results.csv", "w") do io
        avg = 0
        for leaf in nodelist[K]
            
            hist = DRMSMIP.get_history(tree, leaf)
            write(io, "stage, ") + sum(write(io, "$(k), ") for k in 1:K) + write(io, "\n")
            write(io, "B, ") + sum(write(io, string(Bref[leaf,k])*", ") for k in 1:K) + write(io, "\n")
            for l in 1:L
                write(io, "π[$(l)], ") + sum(write(io, string(tree.nodes[id].ξ[l])*", ") for id in hist) + write(io, "\n")
                write(io, "x[$(l)], ") + sum(write(io, string(xref[leaf,k,l])*", ") for k in 1:K) + write(io, "\n")
                write(io, "y[$(l)], ") + sum(write(io, string(yref[leaf,k,l])*", ") for k in 1:K) + write(io, "\n")
            end

            tot = - Bref[leaf,K] - sum( tree.nodes[leaf].ξ[l]*yref[leaf,K,l] for l in 1:L)
            #for k in 1:K
            #    id = hist[k]
            #    tot += sum( tree.nodes[id].cost[l]*yref[leaf,k,l] for l in 1:L) + tree.nodes[id].cost[L+1]*Bref[leaf,k]
            #end
            write(io, "total, " * string(-tot) * "\n")
            write(io, "\n")
            avg += tot
        end
        write(io,"average, "*string(-avg/length(nodelist[K])) * "\n")
        write(io, "objective, "*string(-objective_value(model)))
    end

end