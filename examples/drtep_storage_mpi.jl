using JuMP, Ipopt, GLPK, Gurobi
using DualDecomposition, DRMSMIP
using CSV, SparseArrays

const DD = DualDecomposition
const parallel = DD.parallel

function run_main(foo)
    nStage = parse(Int, foo[1])
    nScenario = parse(Int, foo[2])
    radius = parse(Float64, foo[3])
    filename = foo[4]

    bus_data = CSV.File("./TEP/6-bus/bus_data.csv" ; datarow=2, delim=',')
    branch_data = CSV.File("./TEP/6-bus/branch_data.csv" ; datarow=2, delim=',');

    @assert nScenario <= 36

    nBus = length(bus_data)
    nBranch = length(branch_data)

    discount = 0.95
    LC = 1000 # load curtailment penalty

    generator_efficiency = 0.9
    pump_efficiency = 0.9
    evaporation_remain = 0.9
    r_init = zeros(nBus)
    r_init[3] = 50
    h_max = zeros(nBus)
    h_max[3] = 200

    MAX_LINE_ADDED = 5
    Buses = 1:nBus
    Branches = 1:nBranch

    solMethod = BM.ProximalMethod
    #solMethod = BM.TrustRegionMethod


    from = branch_data.From
    to = branch_data.To
    n0 = branch_data.n0_ij
    gamma = 100 ./ branch_data.Reactance
    fbar = branch_data.barf_ij
    cost = float(branch_data.Cost)

    genMax = bus_data.Gen_Max
    genLevel = bus_data.Gen_Level
    load = bus_data.Load

    inflow = Dict()
    outflow = Dict()
    for bus in Buses
        temp = []
        temp2 = []
        for branch in Branches
            if from[branch] == bus
                push!(temp, branch)
            end
            if to[branch] == bus
                push!(temp2, branch)
            end
        end
        outflow[bus] = temp
        inflow[bus] = temp2
    end
    #@show inflow
    #@show outflow

    function create_tree()::DRMSMIP.DR_Tree
        D_samp = generate_sample(float(bus_data.Load))
        set = DRMSMIP.WassersteinSet(D_samp, radius, DRMSMIP.norm_L1)
        cost_coeff = Float64[]
        for branch in Branches
            for line in 1:MAX_LINE_ADDED
                if line > n0[branch]
                    push!(cost_coeff, cost[branch])# cost for alpha_ijk
                else
                    push!(cost_coeff, 0.0)
                end
            end
        end
        tree = DRMSMIP.DR_Tree([0.0], set, cost_coeff)
        add_nodes!(tree, 1, 1)
        return tree
    end

    function generate_sample(π::Array{Float64})::Array{DRMSMIP.Sample}
        ret = Array{DRMSMIP.Sample}(undef, nScenario)
        multiplier = range(1.0,stop=1.2,length=nScenario)
        for ii in 1:nScenario
            ξ = π * multiplier[ii]
            ret[ii] = DRMSMIP.Sample(ξ, 1/nScenario)
        end
        return ret
    end

    function add_nodes!(tree::DRMSMIP.DR_Tree, id::Int, stage::Int)
        if stage < nStage-1
            for ii in 1:nScenario
                demand = tree.nodes[id].set.samples[ii].ξ
                D_samp = generate_sample(demand)
                set = DRMSMIP.WassersteinSet(D_samp, radius, DRMSMIP.norm_L1)
                cost_coeff = Float64[]
                for branch in Branches
                    for line in 1:MAX_LINE_ADDED
                        push!(cost_coeff, cost[branch] * discount ^ stage )# cost for alpha_ijk
                    end
                end
                for branch in Branches
                    for line in 1:MAX_LINE_ADDED
                        push!(cost_coeff, -cost[branch] * discount ^ stage )# cost for alpha0_ijk
                    end
                end
                for bus in Buses
                    push!(cost_coeff, LC * discount ^ stage)# cost for DC_i
                end
                for bus in Buses
                    push!(cost_coeff, 0.0)# cost for beta_i
                end
                DD.addchild!(tree, id, demand, set, cost_coeff)
                childid = length(tree.nodes)
                add_nodes!(tree, childid, stage+1)
            end
        elseif stage == nStage-1
            for ii in 1:nScenario
                demand = tree.nodes[id].set.samples[ii].ξ
                cost_coeff = Float64[]
                # for branch in Branches
                #     for line in 1:MAX_LINE_ADDED
                #         push!(cost_coeff, cost[branch] * discount ^ stage )# cost for alpha_ijk
                #     end
                # end
                # for branch in Branches
                #     for line in 1:MAX_LINE_ADDED
                #         push!(cost_coeff, -cost[branch] * discount ^ stage )# cost for alpha0_ijk
                #     end
                # end
                for bus in Buses
                    push!(cost_coeff, LC * discount ^ stage)# cost for DC_i
                end
                DD.addchild!(tree, id, demand, nothing, cost_coeff)
            end
        end
    end

    function create_scenario_model(tree::DRMSMIP.DR_Tree, id::Int)
        hist = DD.get_history(tree, id)
        #m = Model(GLPK.Optimizer) 
        m = Model(Gurobi.Optimizer) 
        JuMP.set_optimizer_attribute(m, "OutputFlag", 0)

        @variable(m, alpha[1:nStage,Branches,1:MAX_LINE_ADDED], Bin)
        #@variable(m, 1 >= alpha[1:nStage,Branches,1:MAX_LINE_ADDED] >= 0)
        @variable(m, alpha0[2:nStage,Branches,1:MAX_LINE_ADDED])
        @variable(m, generation[1:nStage,Buses]>=0)
        @variable(m, dc[1:nStage,Buses]>=0)
        @variable(m, π/2 >= theta[1:nStage,Buses] >= -π/2)
        @variable(m, flow[1:nStage,Branches,1:MAX_LINE_ADDED])
        @variable(m, beta[1:nStage,Buses]>=0)
        @variable(m, bG[2:nStage,Buses]>=0)
        @variable(m, bP[2:nStage,Buses]>=0)

        a0 = zeros(nBranch, MAX_LINE_ADDED)
        for branch in Branches, k in 1: MAX_LINE_ADDED
            if k <= n0[branch]
                a0[branch,k] = 1
            end
        end

        @constraints(m, begin
                        [branch in Branches,k in 1:n0[branch]], alpha[1, branch, k] == 1
                        [branch in Branches,k in 2:MAX_LINE_ADDED], alpha[1, branch, k] <= alpha[1, branch, k-1]
                        [bus in Buses], beta[1, bus] == r_init[bus]
                    end)


        for stage = 2:nStage
            demand = tree.nodes[hist[stage]].ξ

            for bus in Buses
                @constraints(m, begin
                    sum(flow[stage, branch, k] for branch in inflow[bus], k in 1:MAX_LINE_ADDED)-sum(flow[stage, branch, k] for branch in outflow[bus], k in 1:MAX_LINE_ADDED) + generation[stage, bus] + dc[stage, bus] + generator_efficiency*bG[stage,bus] - bP[stage,bus]== demand[bus]
                end)
            end

            @constraints(m, begin

                [branch in Branches], sum(flow[stage,branch, k] for k in 1:n0[branch]) - gamma[branch]*n0[branch]*(theta[stage,from[branch]] - theta[stage,to[branch]]) == 0


                [branch in Branches,k in n0[branch]+1:MAX_LINE_ADDED], flow[stage,branch, k] - gamma[branch] * (theta[stage,from[branch]] - theta[stage,to[branch]]) - gamma[branch]*pi * (1-alpha0[stage,branch, k]) <= 0
                [branch in Branches,k in n0[branch]+1:MAX_LINE_ADDED], flow[stage,branch, k] - gamma[branch] * (theta[stage,from[branch]] - theta[stage,to[branch]]) + gamma[branch]*pi * (1-alpha0[stage,branch, k]) >= 0


                [branch in Branches,k in 1:MAX_LINE_ADDED], flow[stage,branch, k] <= fbar[branch] * alpha0[stage,branch, k]
                [branch in Branches,k in 1:MAX_LINE_ADDED], flow[stage,branch, k] >= -fbar[branch] * alpha0[stage,branch, k]

                [bus in Buses], generation[stage,bus] <= genMax[bus]
                [bus in Buses], dc[stage,bus] <= demand[bus]

                [branch in Branches,k in 1:MAX_LINE_ADDED], alpha0[stage,branch, k] == alpha[stage-1,branch, k]
                #[bus in Buses], demand[stage,bus] == bus_data[bus,4]
            end)

            if stage < nStage
                @constraints(m, begin
                    [branch in Branches,k in 1:MAX_LINE_ADDED], alpha[stage,branch, k] >= alpha0[stage,branch, k]
                    [branch in Branches,k in 2:MAX_LINE_ADDED], alpha[stage,branch, k] <= alpha[stage,branch, k-1]
                end)
            end

            @constraints(m, begin
                [bus in Buses], beta[stage,bus] == evaporation_remain * beta[stage-1,bus] + pump_efficiency * bP[stage,bus] - bG[stage,bus]
                [bus in Buses], beta[stage,bus] <= h_max[bus]
                [bus in Buses], bG[stage,bus] <= evaporation_remain * beta[stage-1,bus]
            end)
        end

        @objective(m, Min, 0 )

        # open("model"*string(id)*".lp", "w") do f
        #     print(f, m)
        # end
        return m
    end



    function dual_decomp(tree::DRMSMIP.DR_Tree, nodelist, leafdict)
        # Create DualDecomposition instance.
        #algo = DRMSMIP.DRMS_LagrangeDual(tree, BM.TrustRegionMethod)
        algo = DRMSMIP.DRMS_LagrangeDual(tree, solMethod)

        ##
        parallel.partition(length(nodelist[nStage]))
        ##

        # Add Lagrange dual problem for each scenario s.
        models = Dict{Int,JuMP.Model}(nodelist[nStage][s] => create_scenario_model(tree,nodelist[nStage][s]) for s in parallel.getpartition())
        for s in parallel.getpartition()
            id = nodelist[nStage][s]
            DD.add_block_model!(algo, s, models[id])
        end

        coupling_variables = create_coupling_variables(tree, models, nodelist, leafdict)

        # Set nonanticipativity variables as an array of symbols.
        DD.set_coupling_variables!(algo, coupling_variables)

        bundle_init = DRMSMIP.initialize_bundle(tree, algo)
        return algo, bundle_init
    end

    function create_coupling_variables(tree::DRMSMIP.DR_Tree, models, nodelist, leafdict)
        coupling_variables = Vector{DD.CouplingVariableRef}()
        for s in parallel.getpartition()
            id = nodelist[nStage][s]
            model = models[id]
            hist = DD.get_history(tree, id)

            for stage in 1:nStage-1
                root = hist[stage]
                var_id = 1
                alpha_ref = model[:alpha]
                for branch in Branches
                    for k in 1:MAX_LINE_ADDED
                        push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, var_id], alpha_ref[stage, branch, k]))
                        var_id += 1
                    end
                end
                if stage >= 2
                    alpha0_ref = model[:alpha0]
                    for branch in Branches
                        for k in 1:MAX_LINE_ADDED
                            push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, var_id], alpha0_ref[stage, branch, k]))
                            var_id += 1
                        end
                    end
                    dc_ref = model[:dc]
                    for bus in Buses
                        push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, var_id], dc_ref[stage, bus]))
                        var_id += 1
                    end
                    beta_ref = model[:beta]
                    for bus in Buses
                        push!(coupling_variables, DD.CouplingVariableRef(leafdict[id], [root, var_id], beta_ref[stage, bus]))
                        var_id += 1
                    end
                end
            end
            # dummy coupling variables
            var_id = 1
            dc_ref = model[:dc]
            for bus in Buses
                push!(coupling_variables, DD.CouplingVariableRef(s, [id, var_id], dc_ref[nStage, bus]))
                var_id += 1
            end
        end
        return coupling_variables
    end

    function solve_DD!(algo::DRMSMIP.DRMS_LagrangeDual, bundle_init::Array{Float64,1}, fathom_val::Union{Float64,Nothing}=nothing)
        # Solve the problem with the solver; this solver is for the underlying bundle method.
        # DD.run!(algo, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0), bundle_init)
        DD.run!(algo, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0), bundle_init, fathom_val)
    end

    function set_to_0!(m::JuMP.Model,stage::Int,branch::Int,k::Int)
        alpha = m[:alpha]
        x = alpha[stage,branch,k]
        set_upper_bound(x, 0)
        set_lower_bound(x, 0)
    end

    function set_to_1!(m::JuMP.Model,stage::Int,branch::Int,k::Int)
        alpha = m[:alpha]
        x = alpha[stage,branch,k]
        set_upper_bound(x, 1)
        set_lower_bound(x, 1)
    end

    function unset_bound!(m::JuMP.Model,stage::Int,branch::Int,k::Int)
        alpha = m[:alpha]
        x = alpha[stage,branch,k]
        set_upper_bound(x, 1)
        set_lower_bound(x, 0)
    end

    function set_beta!(m::JuMP.Model,stage::Int,bus::Int,val::Float64)
        beta = m[:beta]
        x = beta[stage,bus]
        set_upper_bound(x, val)
        set_lower_bound(x, val)
    end

    function branch_and_bound(tree::DRMSMIP.DR_Tree, time_limit::Float64)
        
        nodelist = DD.get_stage_id(tree)
        leafdict = DD.leaf2block(nodelist[nStage])
        main_prob = make_main_prob(tree,nodelist)

        parallel.init()
        algo, bundle_init = dual_decomp(tree,nodelist,leafdict)
        feasible_prob = get_problem_copy(algo, nodelist) # list of subproblems
        incumbent = [[algo,-Inf]]
        final_solution = nothing
        tot_time = nothing
        z_hi = nothing
        z_lo = nothing
        if parallel.is_root()
            time_start = time()
            z_hi = Inf
            z_lo = -Inf
            while true
                tot_time = time() - time_start
                if tot_time > time_limit
                    println("maximum time limit reached")
                    parallel.bcast(true)
                    break
                elseif length(incumbent) == 0
                    #parallel.finalize()
                    #return final_solution, feasible_prob, main_prob, nodelist
                    parallel.bcast(true)
                    break
                end
                parallel.bcast(false)
                current_node = pop!(incumbent)
                current = current_node[1]
                solve_DD!(current,bundle_init,z_hi)

                # bounding
                bounding = current.block_model.dual_bound < z_hi
                parallel.bcast(bounding)
                if bounding                
                    alpha_identical, alpha, beta = solution_identical(tree,current,nodelist)
                    #println(alpha)
                    #readline()
                    parallel.bcast((alpha_identical, alpha, beta))
                    if alpha_identical
                        println("alpha identical!")
                        # get feasible objective value and replace z_hi if it is smaller than z_hi
                        z = feasible_objective(tree, feasible_prob, main_prob, alpha, beta, nodelist)
                        z_dl = current.block_model.dual_bound
                        if z < z_hi
                            z_hi = z
                            z_lo = z_dl
                            final_solution = alpha, beta
                            # delete all problems with dual bound greater than z_hi
                            for p in length(incumbent):-1:1
                                lb = incumbent[p][2]
                                if lb > z_hi
                                    deleteat!(incumbent,p)
                                    parallel.bcast(p)
                                end
                            end
                        end
                        parallel.bcast(0)
                    else
                        println("alpha not identical!")
                        # use heuristic to get feasible objective value and replace z_hi if it is smaller than z_hi
                        h_alpha, h_beta = heuristic_solution(tree,alpha,beta)
                        parallel.bcast((h_alpha, h_beta))
                        z = feasible_objective(tree, feasible_prob, main_prob, h_alpha, h_beta, nodelist)
                        z_dl = current.block_model.dual_bound
                        if z < z_hi
                            z_hi = z
                            z_lo = z_dl
                            final_solution = h_alpha, h_beta
                            # delete all problems with dual bound greater than z_hi
                            for p in length(incumbent):-1:1
                                lb = incumbent[p][2]
                                if lb > z_hi
                                    deleteat!(incumbent,p)
                                    parallel.bcast(p)
                                end
                            end
                        end
                        parallel.bcast(0)

                        #branching
                        branching = z > z_dl * (1+1e-6)
                        parallel.bcast(branching)
                        if branching

                            parallel.bcast(z_dl)

                            # select one variable
                            # add bound constraints
                            next0, next1 = branch_variable(tree, current, alpha, nodelist, leafdict)

                            # push to incumbent
                            # always try increasing the lines
                            push!(incumbent, [next0,z_dl])
                            push!(incumbent, [next1,z_dl])
                        end
                    end
                    println("ub: ", z_hi, " lb: ", z_lo)
                end
                println("incumbent: ",length(incumbent))
            end
        else
            while true
                termination = parallel.bcast(nothing)
                if termination
                    break
                end
                current_node = pop!(incumbent)
                current = current_node[1]
                solve_DD!(current,bundle_init,z_hi)

                bounding = parallel.bcast(nothing)
                if bounding
                    solution_identical(tree,current,nodelist)
                    alpha_identical, alpha, beta = parallel.bcast(nothing)

                    if alpha_identical
                        feasible_objective(tree, feasible_prob, main_prob, alpha, beta, nodelist)
                        delid = parallel.bcast(nothing)
                        while delid != 0
                            deleteat!(incumbent,p)
                            delid = parallel.bcast(nothing)
                        end
                    else
                        h_alpha, h_beta = parallel.bcast(nothing)
                        feasible_objective(tree, feasible_prob, main_prob, h_alpha, h_beta, nodelist)
                        delid = parallel.bcast(nothing)
                        while delid != 0
                            deleteat!(incumbent,delid)
                            delid = parallel.bcast(nothing)
                        end
                        branching = parallel.bcast(nothing)
                        if branching
                            z_lb = parallel.bcast(nothing)
                            next0, next1 = branch_variable(tree, current, alpha, nodelist, leafdict)

                            # push to incumbent
                            # always try increasing the lines
                            push!(incumbent, [next0,z_lb])
                            push!(incumbent, [next1,z_lb])
                        end
                    end
                end
            end
        end

        dual_decomp_results(tree, feasible_prob, main_prob, final_solution, nodelist, tot_time, z_hi, z_lo)

        parallel.finalize()
    end

    function get_problem_copy(LD::DRMSMIP.DRMS_LagrangeDual, nodelist)::Dict{Int,JuMP.Model}
        feasible_prob = Dict()
        for s in parallel.getpartition()
            m = copy(LD.block_model.model[s])
            set_optimizer(m, Gurobi.Optimizer)
            JuMP.set_optimizer_attribute(m, "OutputFlag", 0)

            alpha = m[:alpha]
            alpha0 = m[:alpha0]
            dc = m[:dc]
            for branch in Branches
                for k in n0[branch]+1:MAX_LINE_ADDED
                    set_objective_coefficient(m, alpha[1,branch,k], cost[branch])
                end
            end
            for stage in 2:nStage-1
                for branch in Branches
                    for k in 1:MAX_LINE_ADDED
                        set_objective_coefficient(m, alpha[stage,branch,k], cost[branch] * discount ^ (stage-1))
                        set_objective_coefficient(m, alpha0[stage,branch,k], -cost[branch] * discount ^ (stage-1))
                    end
                end
            end
            for stage in 2:nStage
                for bus in Buses
                    set_objective_coefficient(m, dc[stage,bus], LC * discount ^ (stage-1))
                end
            end
            feasible_prob[s] = m
        end
        return feasible_prob
    end

    function make_main_prob(tree::DRMSMIP.DR_Tree, nodelist::Array{Array{Int}})
        K = tree.K
        model = JuMP.Model(Gurobi.Optimizer)
        JuMP.set_optimizer_attribute(model, "OutputFlag", 0)

        @variable(model, P[2:length(tree.nodes)] >= 0)
        @variable(model, w[k=1:K-1, id=nodelist[k+1], s=1:tree.nodes[DD.get_parent(tree,id)].set.N] >= 0)
        
        DRMSMIP.con_E!(tree, model, nodelist)
        DRMSMIP.con_P!(tree, model, nodelist)
        DRMSMIP.con_M!(tree, model, nodelist)
        DRMSMIP.con_N!(tree, model, nodelist)

        @objective(model, Max, 0)
        return model
    end

    function solution_identical(tree::DRMSMIP.DR_Tree, LD::DRMSMIP.DRMS_LagrangeDual, nodelist)
        n = parallel.sum(DD.num_coupling_variables(LD.block_model))
        solutions = Dict{Int,SparseVector{Float64}}()
        for (s,m) in DD.block_model(LD)
            solutions[s] = sparsevec(Dict{Int,Float64}(), n)
        end
        for var in DD.coupling_variables(LD)
            # @assert has_block_model(LD, var.key.block_id)
            solutions[var.key.block_id][DD.index_of_λ(LD, var)] = JuMP.value(var.ref)
        end
        solutions_combined = parallel.combine_dict(solutions)

        if parallel.is_root()
            alpha_average = zeros(length(tree.nodes), nBranch, MAX_LINE_ADDED)
            beta_holder = zeros(length(tree.nodes), nBus)
            for s in 1:length(nodelist[nStage])
                leaf = nodelist[nStage][s]
                hist = DD.get_history(tree, leaf)
                for stage in 1:nStage - 1
                    node = hist[stage]
                    for branch in Branches
                        for k in 1:MAX_LINE_ADDED
                            var_id = (branch-1)*MAX_LINE_ADDED+k
                            i = LD.var_to_index[s,[node,var_id]]
                            alpha_average[node,branch,k] += solutions_combined[s][i]
                        end
                    end
                    if stage>=2
                        for bus in Buses
                            var_id = nBranch*MAX_LINE_ADDED*2+nBus+bus
                            i = LD.var_to_index[s,[node,var_id]]
                            beta_holder[node,bus] = max(beta_holder[node,bus], solutions_combined[s][i])
                        end
                    else
                        for bus in Buses
                            beta_holder[node,bus] = r_init[bus]
                        end
                    end
                end
            end
            alpha_identical = true
            for node in 1:length(tree.nodes)
                stage = DD.get_stage(tree,node)
                if stage < nStage
                    for branch in Branches
                        for k in 1:MAX_LINE_ADDED
                            alpha_average[node,branch,k] = alpha_average[node,branch,k] / nScenario^(nStage-stage)
                            if alpha_average[node,branch,k] < 1e-9
                                alpha_average[node,branch,k] = 0
                            elseif alpha_average[node,branch,k] > 1-1e-9
                                alpha_average[node,branch,k] = 1
                            else
                                alpha_identical = false
                            end
                        end
                    end
                end
            end
            return alpha_identical, alpha_average, beta_holder
        end
    end

    function feasible_objective(tree::DRMSMIP.DR_Tree, feasible_prob, main_prob, alpha, beta, nodelist)
        objectives = Dict{Int,Float64}()
        for s in parallel.getpartition()
            m = feasible_prob[s]
            leaf = nodelist[nStage][s]
            hist = DD.get_history(tree, leaf)
            for stage in 1:nStage - 1
                node = hist[stage]
                for branch in Branches
                    for k in 1:MAX_LINE_ADDED
                        if alpha[node,branch,k] < 1e-9
                            set_to_0!(m,stage,branch,k)
                        else
                            set_to_1!(m,stage,branch,k)
                        end
                    end
                end
                for bus in Buses
                    set_beta!(m,stage,bus,beta[node,bus])
                end
            end
            JuMP.optimize!(m)
            if termination_status(m) != MOI.OPTIMAL
                open("modeltest.lp", "w") do f
                    print(f, m)
                end
                for node in 1:length(tree.nodes)
                    if tree.nodes[node].k < nStage
                        println(node)
                        for k in 1:MAX_LINE_ADDED
                            println(alpha[node,:,k])
                        end
                    end
                end
            end
            objectives[s] = JuMP.objective_value(m)
        end
        objvals_combined = parallel.combine_dict(objectives)

        if parallel.is_root()
            P = main_prob[:P]
            for s in 1:length(nodelist[nStage])
                id = nodelist[nStage][s]
                set_objective_coefficient(main_prob, P[id], objvals_combined[s])
            end
            JuMP.optimize!(main_prob)

            Pref = Dict()

            for id in 2:length(tree.nodes)
                Pref[id] = JuMP.value(P[id])
            end
            # println(objectives,Pref,objective_value(main_prob))
            return objective_value(main_prob)
        end
    end

    function heuristic_solution(tree::DRMSMIP.DR_Tree, alpha, beta)
        h_alpha = zeros(length(tree.nodes), nBranch, MAX_LINE_ADDED)
        for node in 1:length(tree.nodes)
            stage = DD.get_stage(tree,node)
            if stage < nStage
                for branch in Branches
                    for k in 1:MAX_LINE_ADDED
                        if stage >= 2 && h_alpha[tree.nodes[node].parent,branch,k] > 1-1e-9
                            h_alpha[node,branch,k] = 1
                        elseif k>=2 && h_alpha[node,branch,k-1] < 1e-9
                            h_alpha[node,branch,k] = 0
                        else
                            h_alpha[node,branch,k] = round(alpha[node,branch,k])
                        end
                    end
                end
            end
        end
        return h_alpha, beta
    end

    function branch_variable(tree::DRMSMIP.DR_Tree, LD::DRMSMIP.DRMS_LagrangeDual, alpha, nodelist, leafdict)
        if parallel.is_root()
            node, branch, k = choose_branch_variable(tree, alpha)
            parallel.bcast((node,branch,k))
        else
            node, branch, k = parallel.bcast(nothing)
        end
        leaves = DD.get_future(tree,node)
        stage = DD.get_stage(tree,node)
        next0 = hardcopy(LD,nodelist,leafdict)
        next1 = hardcopy(LD,nodelist,leafdict)
        for leaf in leaves
            block_id = leafdict[leaf]
            if block_id in parallel.getpartition()
                m0 = next0.block_model.model[block_id]
                set_to_0!(m0,stage,branch,k)

                m1 = next1.block_model.model[block_id]
                set_to_1!(m1,stage,branch,k)
            end
        end
        return next0, next1
    end


    function hardcopy(LD::DRMSMIP.DRMS_LagrangeDual, nodelist, leafdict)
        #next = DRMSMIP.DRMS_LagrangeDual(tree, BM.TrustRegionMethod)
        next = DRMSMIP.DRMS_LagrangeDual(tree, solMethod)
        models = Dict()
        for s in parallel.getpartition()
            id = nodelist[nStage][s]
            m = copy(LD.block_model.model[s])
            set_optimizer(m, Gurobi.Optimizer)
            JuMP.set_optimizer_attribute(m, "OutputFlag", 0)
            models[id] = m
        end

    for s in parallel.getpartition()
            id = nodelist[nStage][s]
            DD.add_block_model!(next, s, models[id])
        end

        coupling_variables = create_coupling_variables(tree, models, nodelist, leafdict)

        # Set nonanticipativity variables as an array of symbols.
        DD.set_coupling_variables!(next, coupling_variables)
        
        return next
    end


    function choose_branch_variable(tree::DRMSMIP.DR_Tree, alpha)
        for node in 1:length(tree.nodes)
            stage = DD.get_stage(tree,node)
            if stage < nStage
                for branch in Branches
                    for k in 1:MAX_LINE_ADDED
                        if alpha[node,branch,k] > 1e-9 && alpha[node,branch,k] < 1-1e-9 
                            return node, branch, k
                        end
                    end
                end
            end
        end
    end


    function dual_decomp_results(tree::DRMSMIP.DR_Tree, feasible_prob, main_prob, solution, nodelist, time, z_hi, z_lo)
        objectives = Dict{Int,Float64}()
        if parallel.is_root()
            parallel.bcast(solution)
        else
            solution = parallel.bcast(nothing)
        end
        alpha_val = solution[1]
        beta_val = solution[2]

        for s in parallel.getpartition()
            m = feasible_prob[s]
            leaf = nodelist[nStage][s]
            hist = DD.get_history(tree, leaf)
            for stage in 1:nStage - 1
                node = hist[stage]
                for branch in Branches
                    for k in 1:MAX_LINE_ADDED
                        if alpha_val[node,branch,k] < 1e-9
                            set_to_0!(m,stage,branch,k)
                        else
                            set_to_1!(m,stage,branch,k)
                        end
                    end
                end
                for bus in Buses
                    set_beta!(m,stage,bus,beta_val[node,bus])
                end
            end
            JuMP.optimize!(m)
            objectives[s] = JuMP.objective_value(m)
        end
        objvals_combined = parallel.combine_dict(objectives)

        curtailment = Dict{Int,SparseVector{Float64}}()
        for s in parallel.getpartition()
            curtailment[s] = sparsevec(Dict{Int,Float64}(), nStage*nBus)
            m = feasible_prob[s]
            dc_ref = value.(m[:dc])
            for stage in 2:nStage
                for bus in Buses
                    curtailment[s][(stage-1)*nBus + bus] = dc_ref[stage,bus]
                end
            end
        end
        curtailment_combined = parallel.combine_dict(curtailment)


        if parallel.is_root()
            P = main_prob[:P]
            for s in 1:length(nodelist[nStage])
                id = nodelist[nStage][s]
                set_objective_coefficient(main_prob, P[id], objvals_combined[s])
            end
            JuMP.optimize!(main_prob)

            Pref = Dict()

            for id in 2:length(tree.nodes)
                Pref[id] = JuMP.value(P[id])
            end

            tot = 0

            open("./TEPresults/"*filename, "w") do io
                for s in 1:length(nodelist[nStage])
                    leaf = nodelist[nStage][s]
                    
                    hist = DD.get_history(tree, leaf)
                    write(io, "scenario, ") + sum(write(io, string(hist[stage])*", ") for stage in 1:nStage) + write(io, "\n")
                    write(io, "P, , ") + sum(write(io, string(Pref[hist[stage]])*", ") for stage in 2:nStage) + write(io, "\n")
                    write(io, "lines 0, ") + sum(write(io, string(n0[branch])*", ") for branch in Branches) + write(io, "\n")
                    for stage in 1:nStage-1
                        lines = Int64[]
                        node = hist[stage]
                        for branch in Branches
                            push!(lines, sum(alpha_val[node,branch,k] for k in 1:MAX_LINE_ADDED))
                        end
                        write(io, "lines " * string(stage) * ", ") + sum(write(io, string(lines[branch])*", ") for branch in Branches) + write(io, "\n")
                    end
                    for stage in 2:nStage
                        write(io, "curtailments " * string(stage) * ", ") + sum(write(io, string(curtailment_combined[s][(stage-1)*nBus+bus])*", ") for bus in Buses) + write(io, "\n")
                    end
                    for stage in 1:nStage
                        node = hist[stage]
                        write(io, "storage " * string(stage) * ", ") + sum(write(io, string(beta_val[node,bus])*", ") for bus in Buses) + write(io, "\n")
                    end

                    write(io, "total, " * string(objvals_combined[s]) * "\n")
                    write(io, "\n")

                    id = nodelist[nStage][s]
                    tot += objectives[s] * Pref[id]
                end
                write(io, "feas objective, "*string(z_hi)* "\n")
                write(io, "dual objective, "*string(z_lo)* "\n")
                write(io,"time, "*string(time)*"\n")
            end
        end
    end


    # function dual_decomp_results(tree::DRMSMIP.DR_Tree, feasible_prob, main_prob, solution, nodelist, time)
    #     objectives = Dict()
    #     alpha_val = solution[1]
    #     beta_val = solution[2]
    #     for s in 1:length(feasible_prob)
    #         m = feasible_prob[s]
    #         leaf = nodelist[nStage][s]
    #         hist = DD.get_history(tree, leaf)
    #         for stage in 1:nStage - 1
    #             node = hist[stage]
    #             for branch in Branches
    #                 for k in 1:MAX_LINE_ADDED
    #                     if alpha_val[node,branch,k] < 1e-9
    #                         set_to_0!(m,stage,branch,k)
    #                     else
    #                         set_to_1!(m,stage,branch,k)
    #                     end
    #                 end
    #             end
    #             for bus in Buses
    #                 set_beta!(m,stage,bus,beta_val[node,bus])
    #             end
    #         end
    #         JuMP.optimize!(m)
    #         objectives[s] = JuMP.objective_value(m)
    #     end
    #     P = main_prob[:P]
    #     for s in 1:length(feasible_prob)
    #         id = nodelist[nStage][s]
    #         set_objective_coefficient(main_prob, P[id], objectives[s])
    #     end
    #     JuMP.optimize!(main_prob)

    #     Pref = Dict()

    #     for id in 2:length(tree.nodes)
    #         Pref[id] = JuMP.value(P[id])
    #     end

    #     tot = 0

    #     open("./TEPresults/dual_decomp_results.csv", "w") do io
    #         for s in 1:length(nodelist[nStage])
    #             m = feasible_prob[s]
    #             leaf = nodelist[nStage][s]

    #             alpha_ref = value.(m[:alpha])
    #             generation_ref = value.(m[:generation])
    #             dc_ref = value.(m[:dc])
    #             theta_ref = value.(m[:theta])
    #             flow_ref = value.(m[:flow])
    #             beta_ref = value.(m[:beta])
                
    #             hist = DD.get_history(tree, leaf)
    #             write(io, "scenario, ") + sum(write(io, string(hist[stage])*", ") for stage in 1:nStage) + write(io, "\n")
    #             write(io, "P, , ") + sum(write(io, string(Pref[hist[stage]])*", ") for stage in 2:nStage) + write(io, "\n")
    #             write(io, "lines 0, ") + sum(write(io, string(n0[branch])*", ") for branch in Branches) + write(io, "\n")
    #             for stage in 1:nStage-1
    #                 lines = Int64[]
    #                 for branch in Branches
    #                     push!(lines, sum(alpha_ref[stage,branch,k] for k in 1:MAX_LINE_ADDED))
    #                 end
    #                 write(io, "lines " * string(stage) * ", ") + sum(write(io, string(lines[branch])*", ") for branch in Branches) + write(io, "\n")
    #             end
    #             for stage in 2:nStage
    #                 write(io, "curtailments " * string(stage) * ", ") + sum(write(io, string(dc_ref[stage,bus])*", ") for bus in Buses) + write(io, "\n")
    #             end
    #             for stage in 1:nStage
    #                 write(io, "storage " * string(stage) * ", ") + sum(write(io, string(beta_ref[stage,bus])*", ") for bus in Buses) + write(io, "\n")
    #             end

    #             write(io, "total, " * string(objectives[s]) * "\n")
    #             write(io, "\n")

    #             id = nodelist[nStage][s]
    #             tot += objectives[s] * Pref[id]
    #         end
    #         write(io, "objective, "*string(tot)* "\n")
    #         write(io,"time, "*string(time)*"\n")
    #     end
        
    # end

    tree = create_tree()

    branch_and_bound(tree, 3600.0);
end


#run_main([3,2,10.0,"dual_decomp_results_.csv"])
run_main(ARGS)