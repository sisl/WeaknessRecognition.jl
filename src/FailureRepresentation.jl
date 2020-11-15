module FailureRepresentation

include("SystemUnderTest.jl")
include("Adversary.jl")
include("Autoencoder.jl")
include("utils.jl")

using .Adversary
using .Autoencoder
using .SystemUnderTest
using CUDA
using Flux
using Flux: onecold
using Statistics

export
    SystemUnderTest,
    load,
    predict,
    getdata,
    Autoencoder,
    load,
    autoencoder


"""
Load models for black-box system under test `sut` and dataset autoencoder `autoencoder`.
"""
function loadmodels()
    Core.eval(Main, :(import NNlib)) # required for .load
    Core.eval(Main, :(import Flux)) # required for .load
    sut = SystemUnderTest.load("models/sut.bson")
    autoencoder = Autoencoder.load("models/autoencoder.bson")
    encoder = Autoencoder.load("models/encoder.bson", :encoder)
    return sut, autoencoder, encoder
end


"""
Calculate the failure rate of the samples given the `model`, the inputs `X`, and targets `Y`.
"""
failure_rate(model, X, Y) = round(1 - mean(onecold(model(X)) .== onecold(Y)), digits=3)


"""
Main model design iteration loop. See diagram for details.
"""
function design_iteration(; T=1, iters=1, m=1000, k=10)
    sut, autoencoder, encoder = loadmodels()
    _, testdata = SystemUnderTest.getdata()
    X, Y = rand(testdata, m) # `m` samples of testdata
    @info string("Sampled failure rate: ", failure_rate(sut, X, Y))

    data_collection = []

    local adversary
    local mapping

    # model design iteration: loop t
    for t in 1:T
        @info "Model design interation, loop $t"
        # pass X to encoder
        #### encoded_traindata, encoded_testdata = Adversary.splitdata(X, Y, sut, encoder)

        # model evaluation iteration: loop i
        for i in 1:iters
            ## Adversarial step:
            # train adversary using `k` samples of failures from low dimensional representation (linked to true samples)
            # adversary is a classifier that determines if input is a failure or not
            # FIRST STEP: will use supervised learning
            adversary, encoded_testdata, true_testdata = Adversary.train(X, Y, sut, encoder)
            mapping = getmapping(true_testdata) # maps i -> true (x,y)

            ## SUT evaluation
            # pass k samples into SUT to evaluate if they're true failures
            for debug in [true, false]
                X_candidates, Y_candidates = Adversary.select_candidates(adversary, encoded_testdata, mapping; k=k, debug=debug)
                @info "Number of candidate failures: $(length(X_candidates))"

                ## System failure characteristics
                # determine if SUT output is a true failure (`isfailure`)
                # determine `distance` metric of failure (softmax-delta)
                # use those in a reward function that is used by the RL adversary (TODO.)

                ## Run candidate failures on SUT
                failures = 0
                for (x,y) in zip(X_candidates, Y_candidates)
                    sut_prediction::Int = SystemUnderTest.predict(sut, x)[1]
                    true_target::Int = onecold(y)[1]
                    failure::Bool = sut_prediction != true_target
                    @info sut_prediction, true_target, failure, adversary(encoder(x))[1]
                    failures += failure
                end
                failure_percent = failures/length(X_candidates)
                if debug == false
                    push!(data_collection, isnan(failure_percent) ? 0 : failure_percent)
                end
                @info "Percent of candidates true failures: $(failures/length(X_candidates))"
                println("—"^40)
            end
        end
    end

    @info mean(data_collection)

    # use adversary to generate failures.
    # TEST: sample from adversary to get failures
    return X, Y, mapping, adversary
end

end # module
