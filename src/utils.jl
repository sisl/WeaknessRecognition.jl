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


function getmapping(data)
    mapping = Dict()
    for (i, (x,y)) in enumerate(data)
        mapping[i] = (x, y)
    end
    return mapping
end
