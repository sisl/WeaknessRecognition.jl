"""
Failure classifier adversary.

Inputs low dimensional representation, classifies failures based on this.
"""
module Adversary

export accuracy, train, trainandsave, load, getdata

using Base.Iterators: repeated
using BSON: @save, @load
using Colors
using CUDA
using Images
using MLDatasets
using NNlib
using Parameters
using Flux
using Flux.Data: DataLoader
using Flux: @epochs, onehotbatch, onecold, crossentropy, throttle
using Statistics

include("Δsoftmax.jl")

CUDA.allowscalar(false)

@with_kw mutable struct Args
    α::Float64 = 3e-5        # learning rate (default: 3e-4)
    batchsize::Int = 50      # batch size
    epochs::Int = 20         # number of epochs
    device::Function = gpu   # set as gpu if available
    throttle::Int = 1        # throttle print every X seconds
    lowdim::Int = 64         # size of low-dimensional representation
    trainfrac::Float64 = 0.5 # [train | test] fraction
end


global USE_LOW_DIM_REPRESENTATION = true
global USE_BATCHES = false
global USE_ΔSOFTMAX = false


function splitdata(X, Y, sut, encoder, args=Args())
    k_train = floor(Int, size(X,2) * args.trainfrac)
    k_test = size(X,2) - k_train

    X_train = X[:, 1:k_train]
    X_test = X[:, k_train+1:end]

    Y_train_classes = Y[:, 1:k_train]
    Y_test_classes = Y[:, k_train+1:end]

    # evaluate the training data on the SUT to get true target labels
    Y_train = onecold(sut(X_train)) .!= onecold(Y_train_classes)
    # Y_test = fill(NaN, k_test) # no true testing data, we'll have to actually call the SUT after candidate failure selection
    Y_test = onecold(sut(X_test)) .!= onecold(Y_test_classes) # FOR TESTING/DEBUGGING/ANALYSIS PURPOSES

    # Low-dimensional representation of X
    if USE_LOW_DIM_REPRESENTATION
        X̃_train = encoder(X_train)
        X̃_test = encoder(X_test)
    else
        X̃_train = X_train
        X̃_test = X_test
    end

    Δtrain = Δsoftmax(sut, X_train, Y_train_classes)
    Δtest = Δsoftmax(sut, X_test, Y_test_classes)

    X̃_train, Y_train, Δtrain = duplicate_failures(X̃_train, Y_train, Δtrain)

    # split into batches
    if USE_BATCHES
        encoded_traindata = DataLoader(X̃_train, Y_train, batchsize=args.batchsize) #, shuffle=true)
        encoded_testdata = DataLoader(X̃_test, Y_test, batchsize=args.batchsize)
        true_traindata = DataLoader(X_train, Y_train_classes, batchsize=args.batchsize)
        true_testdata = DataLoader(X_test, Y_test_classes, batchsize=args.batchsize)
    else
        if USE_ΔSOFTMAX
            encoded_traindata = DataLoader(X̃_train, Y_train, Δtrain) #, shuffle=true)
            encoded_testdata = DataLoader(X̃_test, Y_test, Δtest)
        else
            encoded_traindata = DataLoader(X̃_train, Y_train) #, shuffle=true)
            encoded_testdata = DataLoader(X̃_test, Y_test)
        end
        true_traindata = DataLoader(X_train, Y_train_classes)
        true_testdata = DataLoader(X_test, Y_test_classes)
    end

    return encoded_traindata, encoded_testdata, true_traindata, true_testdata
end


function convert_data(testdata, sut, encoder)
    x̃ = reduce(hcat, [encoder(x) for (x,_) in testdata])
    ỹ = reduce(vcat, [onecold(sut(x)) .!= onecold(y) for (x,y) in testdata])
    return DataLoader(x̃, ỹ)
end


function num_failures(testdata, sut)
    num = 0
    for (x,y) in testdata
        num += sum(onecold(sut(x)) .!= onecold(y))
    end
    return num
end


function duplicate_failures(X̃_train, Y_train, Δtrain, d=10)
    newX = Matrix{eltype(X̃_train)}(undef, size(X̃_train,1), 0)
    newY = Vector{eltype(Y_train)}(undef, 0)
    newΔ = Vector{eltype(Δtrain)}(undef, 0)
    for (i,y) in enumerate(Y_train)
        if y
            for _ in 1:d
                newX = hcat(newX, X̃_train[:,i])
                newY = vcat(newY, Y_train[i])
                newΔ = vcat(newΔ, Δtrain[i])
            end
        end
        newX = hcat(newX, X̃_train[:,i])
        newY = vcat(newY, Y_train[i])
        newΔ = vcat(newΔ, Δtrain[i])
    end
    return newX, newY, newΔ
end


function buildmodel(args)
    if USE_LOW_DIM_REPRESENTATION
        lowdim = args.lowdim
    else
        lowdim = prod([28,28,1])
    end
    return Chain(
            Dense(lowdim, 2lowdim, relu),
            Dense(2lowdim, lowdim, relu),
            Dense(lowdim, 1, sigmoid))
end


# Flux binarycrossentropy loss fails with use of @.
function bce(ŷ, y; agg=mean, ϵ=Flux.epseltype(ŷ))
    agg(.- Flux.Losses.xlogy.(y, ŷ .+ ϵ) .- Flux.Losses.xlogy.(1 .- y, 1 .- ŷ .+ ϵ))
end

function bce_adversary(ŷ, y; Ω=float32(0.95), agg=mean, ϵ=Flux.epseltype(ŷ))
    agg(.- Flux.Losses.xlogy.(y .* Ω, ŷ .+ ϵ) .- Flux.Losses.xlogy.((1 .- y) .* (1-Ω), 1 .- ŷ .+ ϵ))
end

function bce_adversary_multi(ŷ, y; Ω=float32(2), agg=mean, ϵ=Flux.epseltype(ŷ))
    agg(.- Flux.Losses.xlogy.(y .* Ω, ŷ .+ ϵ) .- Flux.Losses.xlogy.((1 .- y), 1 .- ŷ .+ ϵ))
end

adversarial_loss = bce


"""
Cost function:

\$\\mathcal{J}(\\mathbf{\\hat{y}}, \\mathbf{y}) = \\frac{1}{m}\\sum_{i=1}^m \\mathcal{L}(\\hat{y}_i, y_i)\$

with loss:

\$\\mathcal{L}(\\hat{y}, y) = - y \\log(\\hat y) - (1 - y)\\log(1-\\hat y)\$

\$\\mathcal{L}(\\hat{y}, y) = - y \\log(\\hat y) \\Omega - (1 - y)\\log(1-\\hat y)\$
"""
cost(dataloader, model) = mean(adversarial_loss(model(x), y) for (x,y) in dataloader)


function accuracy(dataloader, model)
    acc = 0
    len = 0
    for (x,y) in dataloader
        acc += sum(predict(model, x)  .== cpu(y)) / size(x,2)
        len += size(x,2)
    end
    return acc/len
end

function failure_accuracy(dataloader, model)
    acc = 0
    len = 0
    for (x,y) in dataloader
        ŷ = predict(model, cpu(x))
        for (i, yᵢ) in enumerate(cpu(y))
            if yᵢ # true failure
                acc += ŷ[i] == cpu(yᵢ) # / size(x,2)
                len += 1
            end
        end
    end
    return acc/len
end


function train(m, X, Y, sut, encoder; kwargs...)
    # initialize model parameters
    args = Args(; kwargs...)

    # load data
    traindata, testdata, true_traindata, true_testdata = splitdata(X, Y, sut, encoder, args)

    # construct model
    if isnothing(m)
        @info "Building new adversarial model."
        m = buildmodel(args)
    else
        @info "Using input adversarial model."
    end
    traindata = args.device.(traindata)
    testdata = args.device.(testdata)
    m = args.device(m)
    # L₂(x, λ=float32(5e-2)) = λ/2 * sum(abs2, x) # L₂ regularization
    # loss(x,y,Δ) = adversarial_loss(m(x), y) + sum(Δ) # Δsoftmax augmentation
    loss(x,y) = adversarial_loss(m(x), y) # + sum(L₂, params(m))

    # callback, optimizer, and training
    callback = throttle(()->@show(cost(traindata, m)), args.throttle)
    opt = ADAM(args.α)

    @info "Training adversary..."
    @epochs args.epochs Flux.train!(loss, params(m), traindata, opt, cb=callback)

    @show accuracy(traindata, m)
    @show accuracy(testdata, m)

    @show failure_accuracy(traindata, m)
    @show failure_accuracy(testdata, m)

    return cpu(m), cpu(testdata), cpu(true_testdata)
end


function predict(m, x, thresh=0.5)
    return cpu(m)(cpu(x)) .>= thresh
end


function select_candidates(model, testdata, mapping; debug=false)
    # Run each X_test datapoint over failure classifier model
    X_candidates = []
    Y_candidates = []
    debug && @warn("select_candidates using true target Y values `debug=true`")
    for (i, (x̃,ỹ)) in enumerate(testdata)
        ŷ = predict(model, x̃) # only select the predictions above X confidence
        for (j, ŷⱼ) in enumerate(ŷ)
            if (!debug && ŷⱼ) || (debug && ỹ[j])
                x, y = mapping[i] # vec(x)
                push!(X_candidates, x[:,j])
                push!(Y_candidates, y[:,j])
            end
        end
    end
    return X_candidates, Y_candidates
end


function rand_select_candidates(testdata, mapping, k)
    X_candidates, Y_candidates = [], []
    if k == 0
        return X_candidates, Y_candidates
    else
        testdata = Flux.Data.DataLoader([d[1] for d in testdata], [d[2][1] for d in testdata])
        rand_idx = rand(1:length(testdata), k)

        for (i, (x̃,ỹ)) in enumerate(testdata)
            for (j, ỹⱼ) in enumerate(ỹ)
                for _ in 1:sum(i .== rand_idx) # counts duplicates from rand
                    x, y = mapping[i] # vec(x)
                    push!(X_candidates, x[:,j])
                    push!(Y_candidates, y[:,j])
                end
            end
        end
        return X_candidates, Y_candidates
    end
end


function trainandsave(X, Y, sut, encoder; kwargs...)
    adversary = train(nothing, X, Y, sut, encoder; kwargs...)
    adversary = cpu(adversary)
    @save "models/adversary.bson" adversary
end


"""
Load pre-trained neural network model.

```julia
using Flux
using CUDA
m = load(\"models/adversary.bson\")
```
"""
function load(model_filename_bson)
    @load model_filename_bson m
    return m
end

end # module
