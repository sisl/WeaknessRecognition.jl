using Revise
using WeaknessRecognition
using Flux: onecold
using ImageView

include("loadmodels.jl")

X, Y, mapping, adversary, metrics, metrics_rand = sampled_validation_iteration(T=10)
