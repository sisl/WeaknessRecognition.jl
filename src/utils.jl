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
