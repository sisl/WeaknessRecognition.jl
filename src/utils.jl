using Flux

"""
Get `m` random samples from a dataset.
"""
function Base.rand(data::Flux.Data.DataLoader, m::Int=1)
    idx = rand(data.indices, m)

    x = selectrand(data.data[1], idx)
    y = selectrand(data.data[2], idx)
    return (x,y)
end

selectrand(data::Union{Matrix, Flux.OneHotMatrix}, idx) = data[:, idx]
selectrand(data::Union{BitArray, Vector}, idx) = data[idx]




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



function getmapping(data)
    mapping = Dict()
    for (i, (x,y)) in enumerate(data)
        mapping[i] = (x, y)
    end
    return mapping
end

#=function getmapping(X, Y, encoder)
    mapping = Dict()
    # X̃ = encoder(X)
    for i in 1:size(X, 2)
        # x̃ = X̃[:,i]
        x = X[:,i]
        y = Y[:,i]
        mapping[i] = (x, y)
    end
    return mapping
end=#