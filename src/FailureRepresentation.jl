module FailureRepresentation

include("SystemUnderTest.jl")
include("Autoencoder.jl")
include("utils.jl")

using .Autoencoder
using .SystemUnderTest
using CUDA
using Flux
using Statistics

export
    SystemUnderTest,
    load,
    predict,
    getdata,
    Autoencoder,
    load,
    autoencoder


function design_iteration(; T=1, iters=1, m=100, k=10)
    sut = SystemUnderTest.load("models/sut.bson")
    autoencoder = Autoencoder.load("models/autoencoder.bson")

    _, testdata = SystemUnderTest.getdata()
    X, Y = rand(testdata, m) # `m` samples of testdata
    @info string("Sampled failure rate: ", 1 - mean(onecold(sut(X)) .== onecold(Y)))

    # model design iteration: loop t
    for t in 1:T
        # pass X to encoder (TODO: strip encoder/decoder from autoencoder)

        # model evaluation iteration: loop i
        for i in 1:iters
            ## Adversarial step:
            # train adversary using `k` samples of failures from low dimensional representation (linked to true samples)
            # adversary is a classifier that determines if input is a failure or not
            # FIRST STEP: will use supervised learning

            ## SUT evaluation
            # pass k samples into SUT to evaluate if they're true failures

            ## System failure characteristics
            # determine if SUT output is a true failure (`isfailure`)
            # determine `distance` metric of failure (softmax-delta)
            # use those in a reward function that is used by the RL adversary (TODO.)
        end
    end

    # use adversary to generate failures.
    # TEST: sample from adversary to get failures
    return X, Y
end

end # module
