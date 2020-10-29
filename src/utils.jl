using Flux

"""
Get `m` random samples from a dataset.
"""
function Base.rand(data::Flux.Data.DataLoader, m::Int=1)
    idx = rand(data.indices, m)
    x = data.data[1][:, idx]
    y = data.data[2][:, idx]
    return (x,y)
end


"""
Δsoftmax distance metric: the difference between the maximum
probability in the softmax of our model output and the probability
associated to the true target value output from our model.

\$\\Delta\\operatorname{softmax}(\\mathbf y, \\mathbf{\\hat y}) =
\\max\\left(\\operatorname{softmax}(\\mathbf{\\hat y})\\right) -
\\operatorname{softmax}(\\mathbf{\\hat y})^{\\left[ \\operatorname{argmax} \\mathbf y \\right]}\$
"""
function Δsoftmax(model, x::Vector, y)
    ŷ = model(x)
    return maximum(softmax(ŷ)) - softmax(ŷ)[argmax(y)]
end

function Δsoftmax(model, x::Matrix, y)
    ŷ = model(x)
    softmaxŷ = softmax(ŷ)
    max_softmax = mapslices(maximum, softmaxŷ, dims=1)'
    true_yᵢ = mapslices(argmax, y, dims=1)
    true_softmax = [softmaxŷ[r,c] for (r,c) in zip(true_yᵢ, 1:size(y,2))]
    return vec(max_softmax - true_softmax)
end