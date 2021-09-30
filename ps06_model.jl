################################################################################
# Author: Hoyoung Yoo, Soojeong Jung
# Date: Septembeer 07, 2021
# Purpose: Econ 899 PS05
################################################################################

#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    N::Int64 = 66 #lifespan
    n::Float64 = 0.011 #population growth
    a_initial::Float64 = 0 #initial asset
    J::Int64 = 46 #retirement
#    θ::Float64 = 0.11 #income tax
    γ::Float64 = 0.42 #weight on consumption
    σ::Float64 = 2 #risk aversion
    Π::Array{Float64, 2} = [0.9261 0.0739; 0.0189 0.9811] #markov process
    Z::Array{Float64, 2} = [3.0 0.5] #state values
    e_dist::Float64 = 0.2037 #ergodic probability of high productivity
    nz::Int64 = length(Z) #number of markov states
#    w::Float64 = 1.05 #wage
#    r::Float64 = 0.05 #interest rate
#    b::Float64 = 0.2 #pension benefit
    δ::Float64 = 0.06 #deprecision
    α::Float64 = 0.36 #capital share
    β::Float64 = 0.97 #discount rate
    a_max::Float64 = 20 #asset upper bound
    a_min::Float64 = 0 #asset lower bound
    na::Int64 = 101 #number of asset grid points
    a_grid::Array{Float64, 1} = collect(range(a_min, length = na, stop = a_max)) #asset grid
    age_eff::Array{Float64, 2} #age-efficiency profile
    μ::Array{Float64,1} #relative sizes of each cohort of age
    ϵ::Float64 = 0.00001 #tolerance
    T::Int64 = 31
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 3} #collapse
    pol_func::Array{Float64, 3}
    pol_ind_func::Array{Float64, 3}
    val_func_R::Array{Float64, 2} #value function for retiree
    val_func_W::Array{Float64, 3} #value function for worker
    pol_func_R::Array{Float64, 2} #policy function for retiree
    pol_ind_R::Array{Int64, 2}
    pol_func_W::Array{Float64, 3} #policy function for worker
    pol_ind_W::Array{Int64, 3}
    lab_func_W::Array{Float64, 3} #labor function for worker
    dist_ss::Array{Float64,3} #steady state distribution of agents over age, productivity, asset
    w::Float64 #wage
    r::Float64 #interest rate
    b::Float64 #pension benefit
    K::Float64 #aggregate capital
    L::Float64 #aggregate labor
    θ::Float64 #income tax
end


#function for initializing model primitives and results
function Initialize(N = 66, n = 0.011)
    μ = zeros(N)
    μ[1] = 1
    for i = 1:N-1
        μ[i+1] = μ[i] / (1 + n)
    end
    μ = μ / sum(μ)
    prim = Primitives(age_eff = age_eff, μ = μ)
    val_func = zeros(prim.na, prim.nz, prim.N)
    pol_func = zeros(prim.na, prim.nz, prim.N)
    pol_ind_func = zeros(prim.na, prim.nz, prim.N)
    val_func_R = zeros(prim.na, prim.N) #initial value function guess
    val_func_W = zeros(prim.na, prim.nz, prim.N) #initial value function guess
    pol_func_R = zeros(prim.na, prim.N) #initial policy function guess
    pol_ind_R = zeros(prim.na, prim.N)
    pol_func_W = zeros(prim.na, prim.nz, prim.N) #initial policy function guess
    pol_ind_W = zeros(prim.na, prim.nz, prim.N)
    lab_func_W = zeros(prim.na, prim.nz, prim.N) #initial labor choice guess
    dist_ss = zeros(prim.na, prim.nz, prim.N) #initialize distribution
    w = 1.05
    r = 0.05
    b = 0.2
    K = 0
    L = 0
    θ = 0.11
    res = Results(val_func, pol_func, pol_ind_func, val_func_R, val_func_W, pol_func_R, pol_ind_R, pol_func_W, pol_ind_W, lab_func_W, dist_ss, w, r, b, K, L, θ) #initialize results struct
    prim, res #return deliverables
end

#utility function
function utility_R(prim::Primitives, c::Float64)
    @unpack σ, γ = prim
    util_R = c^((1-σ)*γ)/(1-σ)
    util_R
end

function utility_W(prim::Primitives, c::Float64, l::Float64)
    @unpack σ, γ = prim
    util_W = ( (c^γ) * ((1-l)^(1-γ)) )^(1-σ)/(1-σ)
    util_W
end

#Backward Induction
function Dynamic_Programming_Retiree(prim::Primitives, res::Results)
    @unpack val_func_R, pol_func_R, pol_ind_R, b, r = res #unpack value function
    @unpack a_grid, σ, γ, N, na, J, a_grid, β = prim #unpack model primitives

    c = (1+r)*a_grid .+ b #column vector consumption
    util_R = c.^((1-σ)*γ)/(1-σ) #column vector utility
    if util_R[1] == -Inf
        util_R[1] = -1e12
    end
    val_func_R[:, N] = util_R
    pol_ind_R[:, N] .= 1

    for j_index = N-1:-1:J
        for a_index = 1:na
            candidate = -1e12

            for ap_index = 1:na
                c = (1 + r) * a_grid[a_index] + b - a_grid[ap_index]

                if c >= 0
                    val =
                        (c^((1 - σ) * γ)) / (1 - σ) +
                        β * val_func_R[ap_index, j_index+1]

                    if val > candidate
                        candidate = val
                        pol_func_R[a_index, j_index] = a_grid[ap_index]
                        pol_ind_R[a_index, j_index] = ap_index
                    end
                end
                if candidate == -1e12
                    pol_ind_R[a_index, j_index] = 1
                end
            end
            val_func_R[a_index, j_index] = candidate
        end
    end


    val_func_R
end

function Optimal_Labor(prim::Primitives, a_index, ap_index, z_index, j_index)
    @unpack γ, σ, age_eff, a_grid, Z = prim
    @unpack r, w, θ = res

    numerator1 = γ*(1-θ)*Z[z_index]*age_eff[j_index]*w
    numerator2 = (1-γ)*((1+r)*a_grid[a_index] - a_grid[ap_index])
    denominator = (1-θ)*w*Z[z_index]*age_eff[j_index]

    opt_labor = (numerator1 - numerator2)/denominator
    if opt_labor > 1
        opt_labor = 1
    elseif opt_labor < 0
        opt_labor = 0
    end
    opt_labor
end


function Dynamic_Programming_Worker(prim::Primitives, res::Results)
    @unpack val_func_W, pol_func_W, lab_func_W, pol_ind_W, val_func_R, r, w, b, θ = res #unpack value function
    @unpack a_grid, σ, γ, N, na, nz, Z, J, a_grid, β, age_eff, Π = prim #unpack model primitives
    for j_index = J-1
        for z_index = 1:nz
            for a_index = 1:na
                candidate = -1e12

                for ap_index = 1:na
                    l = Optimal_Labor(prim, a_index, ap_index, z_index, j_index)
                    c = w*(1-θ)*Z[z_index]*age_eff[j_index]*l + (1+r)*a_grid[a_index] - a_grid[ap_index]

                    if c >= 0
                        val = ( (c^γ) * ((1-l)^(1-γ)) )^(1-σ)/(1-σ) + β*val_func_R[ap_index, J]

                        if val > candidate
                            candidate = val
                            pol_func_W[a_index, z_index, j_index] = a_grid[ap_index]
                            pol_ind_W[a_index, z_index, j_index] = ap_index
                            lab_func_W[a_index, z_index, j_index] = l
                        end
                    end
                end
                val_func_W[a_index, z_index, j_index] = candidate
            end
        end
    end

    for j_index = J-2:-1:1
        for z_index = 1:nz
            for a_index = 1:na
                candidate = -1e12

                for ap_index = 1:na
                    l = Optimal_Labor(prim, a_index, ap_index, z_index, j_index)
                    c = w*(1-θ)*Z[z_index]*age_eff[j_index]*l + (1+r)*a_grid[a_index] - a_grid[ap_index]

                    if c >= 0
                        val = ( (c^γ) * ((1-l)^(1-γ)) )^(1-σ)/(1-σ) + β*sum(Π[z_index,:].*val_func_W[ap_index,:,j_index + 1])

                        if val > candidate
                            candidate = val
                            pol_func_W[a_index, z_index, j_index] = a_grid[ap_index]
                            pol_ind_W[a_index, z_index, j_index] = ap_index
                            lab_func_W[a_index, z_index, j_index] = l
                        end
                    end
                end
                val_func_W[a_index, z_index, j_index] = candidate
            end
        end


    end
    val_func_W
end


function Steady_State_Dist(prim::Primitives, res::Results)
    @unpack N, n, nz, na, a_initial, e_dist, Π, J, μ = prim #unpack model primitives
    @unpack dist_ss, pol_ind_W, pol_ind_R = res #unpack value function

    dist_ss = zeros(na, nz, N)
    dist_ss[1,1,1] = μ[1] * e_dist
    dist_ss[1,2,1] = μ[1] * (1 - e_dist)
    for j_index = 1:J-1
        for z_index = 1:nz
            for a_index = 1:na
                a_next_ind = pol_ind_W[a_index,z_index,j_index]
                dist_ss[a_next_ind,:,j_index+1] += dist_ss[a_index, z_index,j_index] * Π[z_index,:]
            end
        end
        dist_ss[:,:,j_index+1] = dist_ss[:,:,j_index+1] / (1+n)
    end
    dist_ss[:,1,J] = sum(dist_ss[:,:,J], dims=2)
    dist_ss[:,nz,J] .= 0
    for j_index = J:N-1
        for z_index = 1
            for a_index = 1:na
                a_next_ind = pol_ind_R[a_index,j_index]
                dist_ss[a_next_ind,1,j_index+1] +=  dist_ss[a_index,1,j_index]
            end
        end
        dist_ss[:,:,j_index+1] = dist_ss[:,:,j_index+1] / (1+n)
    end
    res.dist_ss = dist_ss
end

function Aggregate_Capital_Labor(prim::Primitives, res::Results)
    @unpack a_grid, age_eff, N, na, nz, J, Z = prim #unpack model primitives
    @unpack dist_ss, lab_func_W = res

    kk = 0
    for j_index = 1:J
        for a_index = 1:na
            for z_index = 1:nz
                kk = kk + dist_ss[a_index,z_index,j_index]*a_grid[a_index]
            end
        end
    end
    for j_index = J+1:N
        for a_index = 1:na
                kk = kk + dist_ss[a_index,1,j_index]*a_grid[a_index]
        end
    end
    ll = 0
    for j_index = 1:J-1
        for a_index = 1:na
            for z_index = 1:nz
                ll += dist_ss[a_index,z_index,j_index]*Z[z_index]*age_eff[j_index]*lab_func_W[a_index,z_index,j_index]
            end
        end
    end
    temp_K = dist_ss.*repeat(a_grid, outer= [1,nz,N])
    K_new = sum(temp_K)

    temp_L = dist_ss[:,:,1:J-1].*repeat(reshape(age_eff, (1,1,45)), inner= [na,nz,1]).*lab_func_W[:,:,1:J-1]
    L_new = sum(temp_L)

    K_new, L_new, kk, ll
end

function Solve_model(prim::Primitives; θ::Float64 = 0.11, zh::Float64=3.0, zl::Float64=0.5, K::Float64=3.5, L::Float64=0.3)
    @unpack ϵ, α, δ, μ, J, N = prim #unpack model primitives

    #Dynamic_Programming_Worker(prim, res)
     #Dynamic_Programming_Retiree(prim, res)
    #Steady_State_Dist(prim,res)
    #(K_now, L_now) = Aggregate_Capital_Labor(prim, res)
#    K = 3.5
#    L = 0.3
#    θ = 0.11

    val_func = zeros(prim.na, prim.nz, prim.N)
    pol_func = zeros(prim.na, prim.nz, prim.N)
    pol_ind_func = zeros(prim.na, prim.nz, prim.N)
    val_func_R = zeros(prim.na, prim.N) #initial value function guess
    val_func_W = zeros(prim.na, prim.nz, prim.N) #initial value function guess
    pol_func_R = zeros(prim.na, prim.N) #initial policy function guess
    pol_ind_R = zeros(prim.na, prim.N)
    pol_func_W = zeros(prim.na, prim.nz, prim.N) #initial policy function guess
    pol_ind_W = zeros(prim.na, prim.nz, prim.N)
    lab_func_W = zeros(prim.na, prim.nz, prim.N) #initial labor choice guess
    dist_ss = zeros(prim.na, prim.nz, prim.N) #initialize distribution

    w = 0.0
    r = 0.0
    b = 0.0
    res = Results(val_func, pol_func, pol_ind_func, val_func_R, val_func_W, pol_func_R, pol_ind_R, pol_func_W, pol_ind_W, lab_func_W, dist_ss, w, r, b, K, L, θ)
    err = 1
    count = 1

    while err > ϵ
        res.w = (1 - α) * (res.K/res.L)^α
        res.r = α * (res.L/res.K)^(1-α) - δ
        res.b = (res.θ*res.w*res.L) / sum( μ[J:N] )
        #println("w: ",res.w, " b ", res.b)
        Dynamic_Programming_Retiree(prim, res)
        Dynamic_Programming_Worker(prim, res)
        Steady_State_Dist(prim,res)
        (K_new, L_new, kk, ll) = Aggregate_Capital_Labor(prim, res)
        L_new = ll
        err = sum( ( [K_new, L_new] - [res.K, res.L] ).^2 )
        res.K = 0.95*res.K + 0.05*K_new
        res.L = 0.95*res.L + 0.05*L_new
        println("Iteration: ", count, " K: ", res.K, " L: ", res.L, " Error: ", err)
        count += 1
    end
    println("w: ", res.w, " r: ", res.r, " b: ", res.b)

    # collapse
    res.val_func = res.val_func_W
    res.val_func[:, 1, J:N] = res.val_func_R[:,J:N]
    res.val_func[:, 2, J:N] = res.val_func_R[:,J:N]

    res.pol_func = res.pol_func_W
    res.pol_func[:, 1, J:N] = res.pol_func_R[:,J:N]
    res.pol_func[:, 2, J:N] = res.pol_func_R[:,J:N]

    res.pol_ind_func = res.pol_ind_W
    res.pol_ind_func[:, 1, J:N] = res.pol_ind_R[:,J:N]
    res.pol_ind_func[:, 2, J:N] = res.pol_ind_R[:,J:N]

    res
end

##############################################################################

mutable struct Results_trans
    val_func::Array{Float64,4} #value functions -- now a 4d object over time to account for transition
    pol_func::Array{Float64,4} #all policy functions over time
    pol_ind_func::Array{Int64,4}
    lab_func::Array{Float64,4} #labor supply function
    stat_dist::Array{Float64, 4} #stationary distribution
    wage::Array{Float64,1} #wage -- now a path
    rate::Array{Float64,1} #interest rate -- also now a path
    ss_b::Array{Float64,1} #path of ss benefits
    cap_agg::Array{Float64,1} #aggregate capital
    lab_agg::Array{Float64,1} #aggregate labor
    theta_path::Array{Float64,1} #path of terms
end

function Backward_Induction(prim::Primitives, res_trans::Results_trans)
    @unpack T, na, nz, N, a_grid, α, μ, δ, J, σ, γ, Z, age_eff, Π, β = prim

    w_next = (1 - α) * (res_trans.cap_agg[T-1]/res_trans.lab_agg[T-1])^α
    res_trans.wage[T-1] = w_next
    r_next = α * (res_trans.lab_agg[T-1]/res_trans.cap_agg[T-1])^(1-α) - δ
    res_trans.rate[T-1] = r_next
    b_next = (res_trans.theta_path[T-1]*w_next*res_trans.lab_agg[T-1]) / sum( μ[J:N] )
    res_trans.ss_b[T-1] = b_next
    θ_next = res_trans.theta_path[T-1]

    for j_index = N
        for a_index = 1:na
            for z_index = 1:nz

                c = (1+r_next)*a_grid[a_index] + b_next #column vector consumption

                res_trans.val_func[a_index, z_index, j_index, T-1] = c^((1-σ)*γ)/(1-σ) #column vector utility
                if res_trans.val_func[a_index, z_index, j_index, T-1] == -Inf
                    res_trans.val_func[a_index, z_index, j_index, T-1] = -1e12
                end
                res_trans.pol_func[a_index, z_index, j_index, T-1] = 0
                res_trans.pol_ind_func[a_index, z_index, j_index, T-1] = 1
                res_trans.lab_func[a_index, z_index, j_index, T-1] = 0
            end
        end
    end

    for j_index = N-1:-1:1
        for a_index = 1:na
            for z_index = 1:nz
                candidate = -1e12

                for ap_index = 1:na

                    if j_index >= J
                        l = 0
                        c = (1+r_next)*a_grid[a_index] - a_grid[ap_index]
                    else
                        l = Optimal_Labor(prim, a_index, ap_index, z_index, j_index)
                        c = w_next*(1-θ_next)*Z[z_index]*age_eff[j_index]*l + (1+r_next)*a_grid[a_index] - a_grid[ap_index]
                    end

                    if c >= 0
                        val = ( (c^γ) * ((1-l)^(1-γ)) )^(1-σ)/(1-σ) + β*sum(Π[z_index,:].*res_trans.val_func[ap_index,:,j_index + 1,T])

                        if val > candidate
                            candidate = val
                            res_trans.pol_func[a_index, z_index, j_index, T-1] = a_grid[ap_index]
                            res_trans.pol_ind_func[a_index, z_index, j_index, T-1] = ap_index
                            res_trans.lab_func[a_index, z_index, j_index, T-1] = l
                        end
                    end
                end
                res_trans.val_func[a_index, z_index, j_index, T-1] = candidate
                if res_trans.pol_func[a_index, z_index, j_index, T-1] == 0
                    res_trans.pol_ind_func[a_index, z_index, j_index, T-1] = 1
                end
            end
        end
    end


    for t_index = T-2:-1:1
    w_next = (1 - α) * (res_trans.cap_agg[t_index]/res_trans.lab_agg[t_index])^α
    res_trans.wage[t_index] = w_next
    r_next = α * (res_trans.lab_agg[t_index]/res_trans.cap_agg[t_index])^(1-α) - δ
    res_trans.rate[t_index] = r_next
    b_next = (res_trans.theta_path[t_index]*w_next*res_trans.lab_agg[t_index]) / sum( μ[J:N] )
    res_trans.ss_b[t_index] = b_next
    θ_next = res_trans.theta_path[t_index]

        for j_index = N
            for a_index = 1:na
                for z_index = 1:nz

                    c = (1+r_next)*a_grid[a_index] + b_next #column vector consumption

                    res_trans.val_func[a_index, z_index, j_index, t_index] = c^((1-σ)*γ)/(1-σ) #column vector utility
                    if res_trans.val_func[a_index, z_index, j_index, t_index] == -Inf
                        res_trans.val_func[a_index, z_index, j_index, t_index] = -1e12
                    end
                    res_trans.pol_func[a_index, z_index, j_index, t_index] = 0
                    res_trans.pol_ind_func[a_index, z_index, j_index, t_index] = 1
                    res_trans.lab_func[a_index, z_index, j_index, t_index] = 0
                end
            end
        end

        for j_index = N-1:-1:1
            for a_index = 1:na
                for z_index = 1:nz
                    candidate = -1e12

                    for ap_index = 1:na

                        if j_index >= J
                            l = 0
                            c = (1+r_next)*a_grid[a_index] - a_grid[ap_index]
                        else
                            l = Optimal_Labor(prim, a_index, ap_index, z_index, j_index)
                            c = w_next*(1-θ_next)*Z[z_index]*age_eff[j_index]*l + (1+r_next)*a_grid[a_index] - a_grid[ap_index]
                        end

                        if c >= 0
                            val = ( (c^γ) * ((1-l)^(1-γ)) )^(1-σ)/(1-σ) + β*sum(Π[z_index,:].*res_trans.val_func[ap_index,:,j_index + 1,t_index + 1])

                            if val > candidate
                                candidate = val
                                res_trans.pol_func[a_index, z_index, j_index, t_index] = a_grid[ap_index]
                                res_trans.pol_ind_func[a_index, z_index, j_index, t_index] = ap_index
                                res_trans.lab_func[a_index, z_index, j_index, t_index] = l
                            end
                        end
                    end
                    res_trans.val_func[a_index, z_index, j_index, t_index] = candidate
                    if res_trans.pol_func[a_index, z_index, j_index, t_index] == 0
                        res_trans.pol_ind_func[a_index, z_index, j_index, t_index] = 1
                    end
                end
            end
        end
    end
end

function Solve_trans(prim::Primitives, res_base::Results, res_noss::Results)
    @unpack T, na, nz, N, a_grid, α, μ, δ, J, σ, γ, Z, age_eff, Π, β = prim

    # make initual guess for the sequences of capital and labor
    cap_agg = collect(range(res_base.K, length = T, stop = res_noss.K))
    lab_agg = collect(range(res_base.L, length = T, stop = res_noss.L))

    theta_path = [res_base.θ; ones(T-1)*res_noss.θ]
    val_func = zeros(na,nz,N,T)
    val_func[:,:,:,1] = res_base.val_func
    val_func[:,:,:,T] = res_noss.val_func
    pol_func = zeros(na,nz,N,T)
    pol_func[:,:,:,1] = res_base.pol_func
    pol_func[:,:,:,T] = res_noss.pol_func
    pol_ind_func = zeros(na,nz,N,T)
    pol_ind_func[:,:,:,1] = res_base.pol_ind_func
    pol_ind_func[:,:,:,T] = res_noss.pol_ind_func
    lab_func = zeros(na,nz,N,T)
    lab_func[:,:,:,1] = res_base.lab_func_W
    lab_func[:,:,:,T] = res_noss.lab_func_W
    stat_dist = zeros(na,nz,N,T)
    stat_dist[:,:,:,1] = res_base.dist_ss
    stat_dist[:,:,:,T] = res_noss.dist_ss
    wage = [res_base.w; zeros(T-2); res_noss.w]
    r = [res_base.r; zeros(T-2); res_noss.r]
    ss_b = [res_base.b; zeros(T-2); res_noss.b]

    res_trans = Results_trans(val_func, pol_func, pol_ind_func, lab_func, stat_dist, wage, r, ss_b, cap_agg, lab_agg, theta_path)

    err = 10
    count = 1

    while err > 1

        Backward_Induction(prim, res_trans)
        Steady_State_Dist_Trans(prim, res_trans)
        K_seq, L_seq = Aggregate_Capital_Labor_Trans(prim, res_trans)

        err = sum((K_seq - res_trans.cap_agg).^2 + (L_seq - res_trans.lab_agg).^2)
        println("Iteration: ", count, ", Error: ", err)

        K_new = 0.1*res_trans.cap_agg + 0.9*K_seq
        L_new = 0.1*res_trans.lab_agg + 0.9*L_seq

        res_trans.cap_agg = K_new
        res_trans.lab_agg = L_new
        count += 1

    end
    res_trans
end

function Steady_State_Dist_Trans(prim::Primitives, res_trans::Results_trans)
    @unpack N, n, nz, na, a_initial, e_dist, Π, J, μ, T = prim #unpack model primitives

    for t_index = 1:T-1

        dist_ss = zeros(na, nz, N)
        dist_ss[1,1,1] = μ[1] * e_dist
        dist_ss[1,2,1] = μ[1] * (1 - e_dist)

        for j_index = 1:J-1
            for z_index = 1:nz
                for a_index = 1:na
                    a_next_ind = res_trans.pol_ind_func[a_index,z_index,j_index,t_index]
                    dist_ss[a_next_ind,:,j_index+1] += dist_ss[a_index, z_index,j_index] * Π[z_index,:]
                end
            end
            dist_ss[:,:,j_index+1] = dist_ss[:,:,j_index+1] / (1+n)
        end
        dist_ss[:,1,J] = sum(dist_ss[:,:,J], dims=2)
        dist_ss[:,nz,J] .= 0
        for j_index = J:N-1
            for z_index = 1
                for a_index = 1:na
                    a_next_ind = res_trans.pol_ind_func[a_index,z_index,j_index,t_index]
                    dist_ss[a_next_ind,1,j_index+1] +=  dist_ss[a_index,1,j_index]
                end
            end
            dist_ss[:,:,j_index+1] = dist_ss[:,:,j_index+1] / (1+n)
        end
        res_trans.stat_dist[:,:,:,t_index+1] = dist_ss
    end
end

function Aggregate_Capital_Labor_Trans(prim::Primitives, res_trans::Results_trans)
    @unpack a_grid, age_eff, N, na, nz, J, Z, T = prim #unpack model primitives

    K_seq = zeros(T)
    L_seq = zeros(T)

    for t_index = 1:T
        for j_index = 1:N
            for a_index = 1:na
                for z_index = 1:nz
                    K_seq[t_index] += res_trans.stat_dist[a_index,z_index,j_index,t_index]*a_grid[a_index]
                end
            end
        end
        for j_index = 1:J-1
            for a_index = 1:na
                for z_index = 1:nz
                    L_seq[t_index] += res_trans.stat_dist[a_index,z_index,j_index,t_index]*Z[z_index]*age_eff[j_index]*res_trans.lab_func[a_index,z_index,j_index,t_index]
                end
            end
        end
    end
    K_seq, L_seq
end

##############################################################################
