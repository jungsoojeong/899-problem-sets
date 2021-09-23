################################################################################
# Author: Hoyoung Yoo, Soojeong Jung
# Date: August 22, 2021
# Purpose: Econ 899 PS02
################################################################################

# cd("C://Users//hoyou//Dropbox//Hoyoung//Computation//PS05")
cd("C://Users//Soo//Desktop//899")

using Parameters, Random, Distributions, Plots, DelimitedFiles
include("ps05_model.jl")

age_file = open("ef.txt")
age_eff = readdlm(age_file)
close(age_file)

# prim = Primitives(age_eff = age_eff) #initialize primtiives

prim, res = Initialize() #initialize primitive and results structs

res_base = Solve_model(prim, res) #solve the model!
res_noss = Solve_model(prim, res; θ = 0.0)


Dynamic_Programming_Retiree(prim, res)
@unpack  a_grid = prim
@unpack  val_func_R, pol_func_R = res

Plots.plot(a_grid, val_func_R[:,50])
Plots.plot(a_grid, pol_func_R[:,50])
Plots.plot!(a_grid, pol_func_R[:,60])

Dynamic_Programming_Worker(prim, res)
@unpack  val_func_W, pol_func_W, lab_func_W = res
Plots.plot(a_grid, val_func_W[:,1,45])
Plots.plot(a_grid, pol_func_W[:,1,45])
Plots.plot(a_grid, lab_func_W[:,1,45])
Plots.plot(a_grid, val_func_W[:,2,20])
Plots.plot(a_grid, lab_func_W[:,2,20])
Plots.plot(a_grid, pol_func_W[:,1,20])
Plots.plot!(a_grid, pol_func_W[:,2,20])

Steady_State_Dist(prim,res)
@unpack  dist_ss = res
Plots.plot(a_grid, sum(sum(dist_ss,dims=3),dims=2)[:,1,1])


#res_norisk = Solve_model(prim, res; zh=0.5, zl=0.5, K=1.5, L=0.3) #baseline
#res_norisk_noss = Solve_model(prim, res; zh=0.5, zl=0.5, K=1.5, L=0.3, theta=0.0) #baseline

@unpack val_func_g, val_func_b, pol_func_g, pol_func_b = res
@unpack k_grid = prim

##############Make plots
#value function
Plots.plot(k_grid, val_func_g, label = "good", title= "Value Function")
Plots.plot!(k_grid, val_func_b, label = "bad", title= "Value Function")
Plots.savefig("02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func_g, label = "good", title= "Policy Functions")
Plots.plot!(k_grid, pol_func_b, label = "bad", title= "Policy Functions")
Plots.savefig("02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func_g).-k_grid
Plots.plot(k_grid, pol_func_δ, title = "Policy Functions Changes")
Plots.savefig("02_Policy_Functions_Changes.png")

println("All done!")
################################
sum(res.val_func_R[:,prim.J:prim.N].*res.dist_ss[:,1,prim.J:prim.N]) + sum(res.val_func_W[:,:,1:prim.J-1].*res.dist_ss[:,:,1:prim.J-1])

prim = Primitives(step_size=0.1, age_eff = age_eff) #model primitives
@unpack na, nz, a_grid,age_eff,N, J = prim #unpack
res_vec = [res_base res_noss] #collect results
for res in res_vec
    println("\n#################################")
    println(res.K) #aggregate capital
    println(res.L) #aggregate labor
    println(res.w) #wage
    println(res.r) #interest rate
    println(res.b) #pension
    println(sum(res.val_func_R[:,prim.J:prim.N].*res.dist_ss[:,1,prim.J:prim.N]) + sum(res.val_func_W[:,:,1:prim.J-1].*res.dist_ss[:,:,1:prim.J-1])) #aggregate welfare

    #computation of coefficient of variation
    dist_ss = ones(na, nz, N)
    dist_ss_sq = ones(na, nz, N)
    a_grid_sq = a_grid.^2

    for a_index = 1:na, z_index = 1:nz, j_index=1:N
        dist_ss[a_index, z_index, j_index] = res.dist_ss[a_index, z_index, j_index]*a_grid[a_index] #update value according to asset amount
        dist_ss_sq[a_index, z_index, j_index] = res.dist_ss[a_index, z_index, j_index]*a_grid_sq[a_index]
    end

    v = sum(dist_ss_sq) - (sum(dist_ss))^2
    cv = (v^.5)/sum(dist_ss)
    println(cv)
    println("#################################\n")
end
