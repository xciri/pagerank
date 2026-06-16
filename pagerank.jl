# PageRank adapted from MATLAB and optimized for Julia
# Author: Aidan Locher

#Imports
using LinearAlgebra
using SparseArrays

#Math
function pagerank(M::AbstractMatrix{T}, max_iterations::Int, d::T) where {T<:AbstractFloat}
    N = size(M, 1)
    @assert size(M, 2) == N #Transition matrix ought to be square

    e = T(1.0e-6)

    #memory allocations baby
    v = fill(1.0 / N, N)
    v_last = Vector{T}(undef, N)

    teleport = (1.0 - d) / N 

    #dangling nodes!
    out_degrees = vec(sum(M, dims=1))
    is_dangling = out_degrees .== 0

    for loop in 1:max_iterations
        v_last .= v

        dangling_contribution = (d * sum(v[is_dangling])) / N

        mul!(v, M, v_last)

        v .= d .* v .+ dangling_contribution .+ teleport

        error = norm(v - v_last, 1)
        if error < e
            return v
        end
    end
    return v
end

#Simulation

function generate_random_web_graph(N::Int, density::Float64)
    println("Generating a random web graph with $N nodes")

    A = sprand(Bool, N, N, density)
    
    for i in 1:N; A[i, i] = false; end

    M = Matrix{Float64}(A)
    out_degrees = sum(M, dims=1)

        for col in 1:N 
            if out_degrees[col] > 0
             M[:, col] ./= out_degrees[col]
        end
    end
return M 
end 

#Pipeline
function run_pipeline()
    println("Validating")

    #Toy network
    M_toy = [0.0 0.0 1.0; 
             0.5 0.0 0.0; 
             0.5 1.0 0.0]
    
    ranks_toy = pagerank(M_toy, 100, 0.85)
    println("Toy network ranks: ", round.(ranks_toy, digits=4))
    println("Probability conservation check (Sum == 1.0): ", sum(ranks_toy))
    println("-"^40)

    N_large = 2000
    M_large = generate_random_web_graph(N_large, 0.005)

    println("Running loop...")
    @time ranks_large = pagerank(M_large, 200, 0.85)

    println("Top 5 highest ranked pages: ", partialsortperm(ranks_large, 1:5, rev=true))
end

#Run
if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline()
end