using Revise
using FailureRepresentation
using Flux: onecold
using ImageView

include("loadmodels.jl")

X, Y, mapping, adversary, metrics, metrics_rand = FailureRepresentation.design_iteration(T=10, seedoffset=1)
