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
