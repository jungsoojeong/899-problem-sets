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
res_trans = Solve_trans(prim, res_base, res_noss)

T = 31
T_grid = collect(1:T)
a = plot(T_grid, res_trans.cap_agg, legend=:none, label="Capital Path", lw=2, xlabel="Time", title="Capital Path")
b = plot(T_grid, res_trans.wage, legend=:none, label="Wage Path", lw=2, xlabel="Time", title="Wage Path")
c = plot(T_grid, res_trans.rate, legend=:none, label="Interest Rate Path", lw=2, xlabel="Time", title="Interest Rate Path")
d = plot(T_grid, res_trans.lab_agg, legend=:none, label="Labor Path", lw=2, xlabel="Time", title="Labor Path")
Plots.plot(a, b, c, d, layout=(2,2))
Plots.savefig("06_paths_ex1.png")
