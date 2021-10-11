################################################################################
# Author: Hoyoung Yoo, Soojeong Jung
# Date: September 23th, 2021
# Purpose: Econ 899 PS06 Transition path
################################################################################

# cd("C://Users//hoyou//Dropbox//Hoyoung//Computation//PS05")
cd("C://Users//Soo//Desktop//899")

using Parameters, Random, Distributions, Plots, DelimitedFiles
include("ps06_model.jl")

age_file = open("ef.txt")
age_eff = readdlm(age_file)
close(age_file)

prim, res = Initialize() #initialize primitive and results structs

# Question 1
res_base = Solve_model(prim) #solve the model!
res_noss = Solve_model(prim; θ = 0.0)

#Question 2
res_trans = Solve_trans(prim, res_base, res_noss, tol = 1e-2)

# plot of paths: borrowed from Garrett
@unpack T = res_trans
T_grid = collect(1:T)
a = plot(T_grid, res_trans.cap_agg, legend=:none, label="Capital Path", lw=2, xlabel="Time", title="Capital Path")
b = plot(T_grid, res_trans.wage, legend=:none, label="Wage Path", lw=2, xlabel="Time", title="Wage Path")
c = plot(T_grid, res_trans.rate, legend=:none, label="Interest Rate Path", lw=2, xlabel="Time", title="Interest Rate Path")
d = plot(T_grid, res_trans.lab_agg, legend=:none, label="Labor Path", lw=2, xlabel="Time", title="Labor Path")
Plots.plot(a, b, c, d, layout=(2,2))
Plots.savefig("06_paths_ex1.png")


###welfare analysis stuff###: borrowed from Garrett

#primitives and such
val_func_start = res_base.val_func
stat_dist_start = res_base.dist_ss
val_func_end = res_trans.val_func[:,:,:,1] #period 1 value function!
@unpack N, n, a_grid, Z, na, nz, γ, σ , μ = prim

#initialize vectors of welfare objects
cons_equiv = zeros(na, nz, N)
pct_favor = 0.0 #population support
pct_favor_age = zeros(N) #percentage support within age groups
avg_ce = zeros(N) #average CE

#loop for computing CE objects for each age
for age = 1:N  #loop over ages
    global pct_favor, pct_favor_age, avg_ce
    for i = 1:na, j = 1:nz #loop over state space
        #compute consumption equivalent
        exp = 1/(γ*(1-σ))
        cons_equiv[i,j,age] = (val_func_end[i,j,age]/val_func_start[i,j,age])^exp -1
        avg_ce[age]+=cons_equiv[i,j,age]*stat_dist_start[i,j,age]/μ[age] #add to average CE for this age group

        #add to measure of percentage favor if bigger than zero
        if cons_equiv[i,j,age]>0
            pct_favor+=stat_dist_start[i,j,age]
            pct_favor_age[age]+=stat_dist_start[i,j,age]/μ[age]
        end
    end
end

#percent in favor of change
println(pct_favor)

#average CE plot
a = plot(collect(1:N), avg_ce, lw=2, title="CE", xlabel = "Age", legend=:none)
b =plot(collect(1:N), pct_favor_age*100, xlabel = "Age", title = "Pct. in Favor", lw=2, legend=:none)
Plots.plot(a, b, layout=(1,2))
Plots.savefig("06_welfare_ex1.png")

############### Exercise 2, delayed announcement ###############
res_trans_delayed = Solve_trans(prim, res_base, res_noss; tol = 1e-2, T = 51, shift_date = 21)


# plot of paths: borrowed from Garrett
@unpack T = res_trans_delayed
T_grid = collect(1:T)
a = plot(T_grid, res_trans_delayed.cap_agg, legend=:none, label="Capital Path", lw=2, xlabel="Time", title="Capital Path")
b = plot(T_grid, res_trans_delayed.wage, legend=:none, label="Wage Path", lw=2, xlabel="Time", title="Wage Path")
c = plot(T_grid, res_trans_delayed.rate, legend=:none, label="Interest Rate Path", lw=2, xlabel="Time", title="Interest Rate Path")
d = plot(T_grid, res_trans_delayed.lab_agg, legend=:none, label="Labor Path", lw=2, xlabel="Time", title="Labor Path")
Plots.plot(a, b, c, d, layout=(2,2))
Plots.savefig("06_paths_ex2.png")


###welfare analysis stuff###: borrowed from Garrett

#primitives and such
val_func_start = res_base.val_func
stat_dist_start = res_base.dist_ss
val_func_end = res_trans_delayed.val_func[:,:,:,1] #period 1 value function!
@unpack N, n, a_grid, Z, na, nz, γ, σ , μ = prim

#initialize vectors of welfare objects
cons_equiv = zeros(na, nz, N)
pct_favor = 0.0 #population support
pct_favor_age = zeros(N) #percentage support within age groups
avg_ce = zeros(N) #average CE

#loop for computing CE objects for each age
for age = 1:N  #loop over ages
    global pct_favor, pct_favor_age, avg_ce
    for i = 1:na, j = 1:nz #loop over state space
        #compute consumption equivalent
        exp = 1/(γ*(1-σ))
        cons_equiv[i,j,age] = (val_func_end[i,j,age]/val_func_start[i,j,age])^exp -1
        avg_ce[age]+=cons_equiv[i,j,age]*stat_dist_start[i,j,age]/μ[age] #add to average CE for this age group

        #add to measure of percentage favor if bigger than zero
        if cons_equiv[i,j,age]>0
            pct_favor+=stat_dist_start[i,j,age]
            pct_favor_age[age]+=stat_dist_start[i,j,age]/μ[age]
        end
    end
end

#percent in favor of change
println(pct_favor)

#average CE plot
a = plot(collect(1:N), avg_ce, lw=2, title="CE", xlabel = "Age", legend=:none)
b =plot(collect(1:N), pct_favor_age*100, xlabel = "Age", title = "Pct. in Favor", lw=2, legend=:none)
Plots.plot(a, b, layout=(1,2))
Plots.savefig("06_welfare_ex2.png")
