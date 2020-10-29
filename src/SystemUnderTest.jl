"""
Black-box system under test (SUT). In this case,
we use an MNIST neural network classifier.
"""
module SystemUnderTest

export accuracy, train, trainandsave, load, getdata

using Base.Iterators: repeated
using BSON: @save, @load
using Colors
using CUDA
using Images
using MLDatasets
using Parameters
using Flux
using Flux.Data: DataLoader
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, throttle
using Statistics


@with_kw mutable struct Args
    α::Float64 = 3e-4      # learning rate
    batchsize::Int = 1024  # batch size
    epochs::Int = 20       # number of epochs
    device::Function = gpu # set as gpu if available
    throttle::Int = 1      # throttle print every X seconds
end


function getdata(args=Args())
    # load dataset
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # reshape data to flatten each image into a vector
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # one-hot encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # split into batches
    traindata = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    testdata = DataLoader(xtest, ytest, batchsize=args.batchsize)

    return traindata, testdata
end


function buildmodel(; imgsize=(28,28,1), nclasses=10)
    return Chain(
            Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end


"""
Cost function:

\$\\mathcal{J}(\\mathbf{\\hat{y}}, \\mathbf{y}) = \\frac{1}{m}\\sum \\mathcal{L}(\\hat{y}, y)\$

with loss:

\$\\mathcal{L}(\\hat{y}, y) = -\\frac{1}{n}\\sum_{i=1}^n y \\left(\\hat{y} - \\log\\left(\\sum e^{\\hat{y}}\\right)\\right)\$
"""
cost(dataloader, model) = mean(logitcrossentropy(model(x), y) for (x,y) in dataloader)


function accuracy(dataloader, model)
    acc = 0
    for (x,y) in dataloader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x,2)
    end
    return acc/length(dataloader)
end


function train(; kwargs...)
    # initialize model parameters
    args = Args(; kwargs...)

    # load data
    traindata, testdata = getdata(args)

    # construct model
    m = buildmodel()
    traindata = args.device.(traindata)
    testdata = args.device.(testdata)
    m = args.device(m)
    loss(x,y) = logitcrossentropy(m(x), y)

    # callback, optimizer, and training
    callback = throttle(()->@show(cost(traindata, m)), args.throttle)
    opt = ADAM(args.α)

    @epochs args.epochs Flux.train!(loss, params(m), traindata, opt, cb=callback)

    @show accuracy(traindata, m)
    @show accuracy(testdata, m)

    return m, traindata, testdata
end


function predict(m, x)
    return onecold(cpu(m(x)))
end


function trainandsave(; kwargs...)
    m, traindata, testdata = train(; kwargs...)
    @save "sut.bson" m
end


"""
Load pre-trained neural network model.

```julia
using Flux
using CUDA
m = load(\"models/sut.bson\")
```
"""
function load(model_filename_bson)
    @load model_filename_bson m
    return m
end

end # module
