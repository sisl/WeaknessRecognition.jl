module FailureRepresentation

include("SystemUnderTest.jl")
include("Adversary.jl")
include("Autoencoder.jl")
include("utils.jl")

using .Adversary
using .Autoencoder
using .SystemUnderTest
using BSON: @save
using CUDA
using Flux
using Flux: onecold
using Parameters
using Random
using Statistics

export
    SystemUnderTest,
    load,
    predict,
    getdata,
    Autoencoder,
    load,
    autoencoder


@with_kw mutable struct Metrics
    precision = [] # of examples recognized as failures, what percent ARE failures?
    recall = [] # of all examples that are failures, what percent are correctly classified?
end


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
function design_iteration(; T=1, iters=1, m=500, k=NaN, seedoffset=1)
    sut, autoencoder, encoder = loadmodels()
    _, testdata = SystemUnderTest.getdata()

    metrics = Metrics()
    metrics_rand = Metrics()
    local adversary = nothing # initialize to "no model"
    local mapping
    local X
    local Y

    # model design iteration: loop t
    for t in 1:T
        @info "Model design interation, loop $t"
        Random.seed!(t*seedoffset)
        X, Y = rand(testdata, m) # `m` samples of testdata
        @info string("Sampled failure rate: ", failure_rate(sut, X, Y))

        # model evaluation iteration: loop i
        for i in 1:iters
            ## Adversarial step:
            # train adversary using `k` samples of failures from low dimensional representation (linked to true samples)
            # adversary is a classifier that determines if input is a failure or not
            # FIRST STEP: will use supervised learning
            adversary, encoded_testdata, true_testdata = Adversary.train(adversary, X, Y, sut, encoder)
            mapping = getmapping(true_testdata) # maps i -> true (x,y)

            # Save out adversary for analysis
            @save "models/adversary_$t.bson" adversary

            ## SUT evaluation
            # pass k samples into SUT to evaluate if they're true failures
            last_num_cands = 0
            num_actual_failures = 0
            for selection_mode in [:debug, :adversary, :random]
                if selection_mode == :debug
                    X_candidates, Y_candidates = Adversary.select_candidates(adversary, encoded_testdata, mapping; k=k, debug=true)
                    num_actual_failures = length(X_candidates)
                elseif selection_mode == :adversary
                    X_candidates, Y_candidates = Adversary.select_candidates(adversary, encoded_testdata, mapping; k=k)
                elseif selection_mode == :random
                    X_candidates, Y_candidates = Adversary.rand_select_candidates(encoded_testdata, mapping, last_num_cands)
                else
                    error("No selection mode: $selection_mode")
                end

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
                precision = failures/length(X_candidates)
                precision = isnan(precision) ? 0 : precision
                recall = failures/num_actual_failures
                recall = isnan(recall) ? 0 : recall
                if selection_mode == :adversary
                    push!(metrics.precision, precision)
                    push!(metrics.recall, recall)
                    last_num_cands = length(X_candidates)
                elseif selection_mode == :random
                    # Random sampling of encoded_testdata, using the same number of selected candidates as the adversary
                    push!(metrics_rand.precision, precision)
                    push!(metrics_rand.recall, recall)
                end
                @info "Precision: $precision"
                @info "Recall: $recall"
                println("—"^40)
            end
        end
    end

    println()
    println()
    @info "Random selection precision: $(mean(metrics_rand.precision))"
    @info "Adversary selection precision: $(mean(metrics.precision))"
    println("—"^40)
    @info "Random selection recall: $(mean(metrics_rand.recall))"
    @info "Adversary selection recall: $(mean(metrics.recall))"

    # use adversary to generate failures.
    # TEST: sample from adversary to get failures
    return X, Y, mapping, adversary, metrics, metrics_rand
end

end # module
