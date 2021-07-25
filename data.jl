using SparseArrays, DataFrames, CSV, MAT, DelimitedFiles, Random, JuMP, MathOptInterface, GLPK, ECOS

function save_sparse_vector(name, var, M)
    CSV.write(name*".txt", DataFrame(vector = [var], row = [length(M)], col = [1]), append = true, writeheader = true)
    I, V = findnz(M)
    df = DataFrame([:I => I, :V => V])
    CSV.write(name*".txt", df, append = true, writeheader = true)
end

function save_sparse_matrix(name, var, M)
    m, n = size(M)
    CSV.write(name*".txt", DataFrame(matrix = [var], row = [m], col = [n]), append = true, writeheader = true)
    I, J, V = findnz(M)
    df = DataFrame([:I => I, :J => J, :V => V])
    CSV.write(name*".txt", df, append = true, writeheader = true)
end

function save_sparse_data(file, data)
    for (key, value) in data
        if typeof(value) == Core.Float64 || typeof(value) == Core.Int64
            save_sparse_vector(file, key, sparsevec([value]))
        elseif typeof(value) == SparseArrays.SparseVector{Float64, Int64}         
            save_sparse_vector(file, key, value)
        else
            save_sparse_matrix(file, key, sparse(value))
        end
    end
    
    matwrite(file*".mat", data; compress = true)
end

function generate_data(file, k)    
    data = matread(file)
    name = collect(keys(data))[1]
    data = data[name]
    S = sparse(data["S"])
    m, n = size(S)

    L = spzeros(n, k)
    U = spzeros(n, k)

    indices = randperm(n)
    core = indices[1:Int(floor(n/10))]
    inside = indices[1:Int(floor(n/2))]
    outside = indices[Int(floor(n/2)+1):n]

    counter = 1

    while counter <= k
        l1 = spzeros(n, 1)
        u1 = spzeros(n, 1)

        bound1 = sprandn(Int(floor(n/2)), 1, 4/sqrt(n))
        bound2 = sprandn(Int(floor(n/2)), 1, 4/sqrt(n))
        l1[inside] = min.(0.0, bound1, bound2)
        u1[inside] = max.(0.0, bound1, bound2)
        
        l1[core] .= -1
        u1[core] .= +1

        warmup_model = Model(GLPK.Optimizer)

        @variable(warmup_model, v[1:n])
        @constraint(warmup_model, u1 .>= v .>= l1)
        @constraint(warmup_model, S * v .== spzeros(m))
        @objective(warmup_model, Min, sum(randn(n, 1) .* v))

        optimize!(warmup_model)

        if termination_status(warmup_model) != MathOptInterface.OPTIMAL
            continue
        end

        v = sparsevec(value.(v))

        l1[core] = v[core] - 0.5*abs.(v[core]) .* randexp(Int(floor(n/10)), 1)
        u1[core] = v[core] + 0.5*abs.(v[core]) .* randexp(Int(floor(n/10)), 1)
        
        if all(l1[core] .<= 0.0) && all(u1[core] .>= 0.0)
            continue
        end

        bound1 = sprandn(Int(ceil(n/2)), 1, 4/sqrt(n))
        bound2 = sprandn(Int(ceil(n/2)), 1, 4/sqrt(n))
        l1[outside] = min.(0.0, bound1, bound2)
        u1[outside] = max.(0.0, bound1, bound2)

        L[:, counter] = l1
        U[:, counter] = u1

        println("counter = ", counter)
        counter += 1
    end

    l1 = L[:,1]
    u1 = U[:,1]

    data = Dict{String, Any}(
        "S" => S,
        "l1" => l1,
        "u1" => u1)

    save_sparse_data(name*"round1and2", data)

    data = Dict{String, Any}(
        "S" => S,
        "L" => L,
        "U" => U)

    save_sparse_data(name*"round3", data)

    data = Dict{String, Any}(
        "S" => S,
        "lambda" => 1.5*n/k,
        "L" => L,
        "U" => U)

    save_sparse_data(name*"round4", data)

    data = Dict{String, Any}(
        "S" => S,
        "K" => Int(floor(k/5)),
        "L" => L,
        "U" => U)

    save_sparse_data(name*"round5", data)
    
    selection = spzeros(n, 4*k)
    for i in 1:4
        selection[inside, ((i-1)*k+1):(i*k)] = sprand(Bool, Int(floor(n/2)), k, 0.2*i)
    end

    Ltilde = [L L L L] .* selection
    Utilde = [U U U U] .* selection

    judge = Dict{String, Any}(
        "Ltilde" => Ltilde,
        "Utilde" => Utilde)

    save_sparse_data(name*"judge5", judge)

    novice_model = Model(GLPK.Optimizer)

    @variable(novice_model, v[1:n])
    @variable(novice_model, v_abs[1:n])
    @constraint(novice_model, u1 .>= v .>= l1)
    @constraint(novice_model, v_abs .>= v)
    @constraint(novice_model, v_abs .>= -v)
    @constraint(novice_model, S * v .== spzeros(m))
    @objective(novice_model, Min, sum(v_abs[i] for i in 1:n))

    optimize!(novice_model)
    
    @show termination_status(novice_model)
    
    v = sparsevec(value.(v))

    solution = Dict{String, Any}(
        "v" => v)

    save_sparse_data(name*"solution1and2", solution)
    writedlm(name*"solution1and2.csv", v, ',')

    @show nnz(v)
    
    if k <= 50
        expert_model = Model(ECOS.Optimizer)

        @variable(expert_model, V[1:(k * n)])
        @variable(expert_model, v_abs[1:n])
        @constraint(expert_model, vec(U) .>= V .>= vec(L))
        for i in 1:k
            @constraint(expert_model, S * V[i*n-n+1 : i*n] .== spzeros(m))
        end
        for i in 1:n
            @constraint(expert_model, [v_abs[i]; V[[j*n-n+i for j in 1:k]]] in SecondOrderCone())
        end
        @objective(expert_model, Min, sum(v_abs[i] for i in 1:n))

        optimize!(expert_model)

        @show termination_status(expert_model)

        V = sparse(reshape(value.(V), (n, k)))

        solution = Dict{String, Any}(
            "V" => V)

        save_sparse_data(name*"solution3and4and5", solution)
        writedlm(name*"solution3and4and5.csv", V, ',')
    end
end

function test_data(file, name)
    data = matread(file)
    S = sparse(data["S"])
    L = sparse(data["L"])
    U = sparse(data["U"])
    m, n = size(S)
    n, k = size(L)

    l1 = L[:,1]
    u1 = U[:,1]

    data = Dict{String, Any}(
        "S" => S,
        "K" => Int(floor(k/5)),
        "L" => L,
        "U" => U)

    save_sparse_data(name*"round5", data)
    
    selection = spzeros(n, 4*k)
    for i in 1:4
        selection[:, ((i-1)*k+1):(i*k)] = sprand(Bool, n, k, 0.2*i)
    end

    Ltilde = [L L L L] .* selection
    Utilde = [U U U U] .* selection

    judge = Dict{String, Any}(
        "Ltilde" => Ltilde,
        "Utilde" => Utilde)

    save_sparse_data(name*"judge5", judge)

    novice_model = Model(GLPK.Optimizer)

    @variable(novice_model, v[1:n])
    @variable(novice_model, v_abs[1:n])
    @constraint(novice_model, u1 .>= v .>= l1)
    @constraint(novice_model, v_abs .>= v)
    @constraint(novice_model, v_abs .>= -v)
    @constraint(novice_model, S * v .== spzeros(m))
    @objective(novice_model, Min, sum(v_abs[i] for i in 1:n))

    optimize!(novice_model)
    
    @show termination_status(novice_model)
    
    v = sparsevec(value.(v))

    solution = Dict{String, Any}(
        "v" => v)

    save_sparse_data(name*"solution1and2", solution)
    writedlm(name*"solution1and2.csv", v, ',')

    @show nnz(v)
    
    if k <= 50
        expert_model = Model(ECOS.Optimizer)

        @variable(expert_model, V[1:(k * n)])
        @variable(expert_model, v_abs[1:n])
        @constraint(expert_model, vec(U) .>= V .>= vec(L))
        for i in 1:k
            @constraint(expert_model, S * V[i*n-n+1 : i*n] .== spzeros(m))
        end
        for i in 1:n
            @constraint(expert_model, [v_abs[i]; V[[j*n-n+i for j in 1:k]]] in SecondOrderCone())
        end
        @objective(expert_model, Min, sum(v_abs[i] for i in 1:n))

        optimize!(expert_model)

        @show termination_status(expert_model)

        V = sparse(reshape(value.(V), (n, k)))

        solution = Dict{String, Any}(
            "V" => V)

        save_sparse_data(name*"solution3and4and5", solution)
        writedlm(name*"solution3and4and5.csv", V, ',')
    end
end

generate_data("data/ecoli_core_model.mat", 20)

generate_data("data/iYS1720.mat", 20)

generate_data("data/iCHOv1.mat", 30)

generate_data("data/iLB1027_lipid.mat", 50)

generate_data("data/Mouse-GEM.mat", 50)

generate_data("data/Recon3D_301.mat", 100)

generate_data("data/universal_model.mat", 200)