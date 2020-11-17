module Autoencoder

export trainandsave, load

using Base.Iterators: partition
using BSON
using BSON: @save
using CUDA
using Images
using NNlib
using Parameters
using Flux
using Flux.Data.MNIST
using Flux: @epochs, mse, throttle
using Statistics


@with_kw mutable struct Args
    α::Float64 = 1e-3      # learning rate
    epochs::Int = 20       # number of epochs
    N::Int = 64            # size of the encoding (i.e. hidden layer)
    batchsize::Int = 100   # batch size for training
    num::Int = 20          # number of random digits in the sample image (UNUSED)
    throttle::Int = 1      # throttle timeout (called once every X seconds)
end


global X = MNIST.images(:test) # train using the same validation dataset


function get_processed_data(args)
    # load images and convert image of type RBG to Float
    imgs = channelview.(X)
    # partition into batches of size `batchsize`
    batches = partition(imgs, args.batchsize)
    traindata = [float(hcat(vec.(imgs)...)) for imgs in batches]
    return gpu.(traindata)
end


function train(; kwargs...)
    args = Args(; kwargs...)

    traindata = get_processed_data(args)

    @info "Constructing model..."

    # You can try to make the encoder/decoder network larger
    # Also, the output of encder is a coding of the given input.
    # In this case, the input dimension is 28^2 and the output dimension of the
    # encoder is 32. This implies that the coding is a compressed representation.
    # We can make lossy compression via this `encoder`.
    intermediate = 28^2 ÷ 2
    encoder = Chain(
        Dense(28^2, intermediate, leakyrelu) |> gpu,
        Dense(intermediate, args.N, leakyrelu) |> gpu)
    decoder = Chain(
        Dense(args.N, intermediate, leakyrelu) |> gpu,
        Dense(intermediate, 28^2, leakyrelu) |> gpu)

    # define main model as a Chain of encoder and decoder models
    m = Chain(encoder, decoder)

    @info "Training model..."
    loss(x) = mse(m(x), x)

    # callback, optimizer, and training
    callback = throttle(() -> @show(mean(loss.(traindata))), args.throttle)
    opt = ADAM(args.α)

    @epochs args.epochs Flux.train!(loss, params(m), zip(traindata), opt, cb=callback)

    return m, encoder, decoder
end


img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

"""
Sample a compressed image from the model.

```julia
sample(m, 20)
```
"""
function sample(m, num=1)
    # convert image of type RGB to Float
    imgs = channelview.(X)
    # number of random digits (truth)
    before = [imgs[i] for i in rand(1:length(imgs), num)]
    # after applying autoencoder to `before` input image
    after = img.(map(x->cpu(m)(float(vec(x))), before))
    # stack `before` and `after` images them all together
    Gray.(hcat(vcat.(before, after)...))
end


function trainandsave(; kwargs...)
    m, encoder, decoder = train(; kwargs...)
    m = cpu(m)
    @save "models/autoencoder.bson" m
    encoder = cpu(encoder)
    @save "models/encoder.bson" encoder
    decoder = cpu(decoder)
    @save "models/decoder.bson" decoder
end


function load(model_filename_bson, sym=:m)
    model = BSON.load(model_filename_bson)[sym]
    return model
end

end # module