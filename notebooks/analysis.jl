### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
using FailureRepresentation

# ╔═╡ 5c706e30-19b1-11eb-18ad-5190eca820b1
begin
	using BSON
	using BSON: load, @load
	using CUDA
	using FailureRepresentation.SystemUnderTest
	using FailureRepresentation.Autoencoder
	using Flux
	using Flux: onecold
	using Flux.Data: DataLoader
	using NNlib
	Core.eval(Main, :(import NNlib)) # required for .load
	Core.eval(Main, :(import Flux)) # required for .load
end

# ╔═╡ 330f1d40-282d-11eb-29e0-f3affbba2711
using PlutoUI

# ╔═╡ d4fcfdae-1a14-11eb-0a76-5dfe76d1754a
using Plots; plotlyjs()

# ╔═╡ 2c1e5cc0-27c5-11eb-1e52-6121d656b973
using PyPlot, Seaborn

# ╔═╡ 27c76940-27c6-11eb-0ba2-c9213d619fb4
using MLBase

# ╔═╡ 5455e46e-27d3-11eb-1478-0f9879fc2215
using ScikitLearn

# ╔═╡ 4147f9a0-27d7-11eb-32de-a9d04fc45211
using ColorSchemes

# ╔═╡ 54164520-19b6-11eb-376c-b5246d05c3a6
MODEL = BSON.load("../models/sut.bson")[:m]

# ╔═╡ 30bd5680-19b2-11eb-3ed2-81593376e639
sut = SystemUnderTest.load("../models/sut.bson")

# ╔═╡ 1732d5e0-19b3-11eb-3b9d-8b5b61fc38ae
traindata, testdata = SystemUnderTest.getdata();

# ╔═╡ 1e4525a0-27a4-11eb-0422-9d0d7dc3dd9c
testdata

# ╔═╡ 24ef3930-19b3-11eb-0d9d-e3a0733ba1da
X, Y = rand(testdata, 3)

# ╔═╡ c3996450-1a14-11eb-2b50-934336a06d5a
Xc, Yc = deepcopy(X), deepcopy(Y);

# ╔═╡ b3418cb2-19b3-11eb-25b4-4b48a12ad82e
hcat([Autoencoder.img(X[:,i])' for i in 1:3])

# ╔═╡ c2f9dec0-27a3-11eb-0b14-75176c151264
hasfailure = onecold(sut(X)) .!= onecold(Y)

# ╔═╡ 8becbb82-19b3-11eb-2c35-b17e8dee1ed2
onecold(sut(X)) .- 1, onecold(Y) .- 1 # notice off-by-one

# ╔═╡ 708c22b0-19b1-11eb-3ee7-f32bf48ebf1f
autoencoder = Autoencoder.load("../models/autoencoder.bson")

# ╔═╡ 1cef6c80-19b5-11eb-1f69-7356f410b1fc
hcat([Autoencoder.img(autoencoder(X[:,i]))' for i in 1:3])

# ╔═╡ 3cf13db0-19b5-11eb-081b-ed8141b6b8d7
onecold(sut(autoencoder(X))) .- 1, onecold(Y) .- 1 # notice off-by-one

# ╔═╡ f4fbfc40-19b2-11eb-0ee1-0b3e9ece6008
Autoencoder.sample(autoencoder, 5)

# ╔═╡ 8cde0240-1a14-11eb-0274-090f1dcdeb0d
softmax(sut(X[:,1]))

# ╔═╡ b6901270-1a16-11eb-09a2-c1073bff3f8f
md"""
$$\Delta\operatorname{softmax}(\mathbf y, \mathbf{\hat y}) = \max\left(\operatorname{softmax}(\mathbf{\hat y})\right) - \operatorname{softmax}(\mathbf{\hat y})^{\left[ \operatorname{argmax} \mathbf y \right]}$$
"""

# ╔═╡ 5d045420-2833-11eb-1f67-59da9ee8ef92
md"""
-  $$\Delta\operatorname{softmax} \in [0, 1]$$
- When $$\Delta\operatorname{softmax}$$ is large, severe failure
- When $$\Delta\operatorname{softmax}$$ is small, likely failure
- When $$\Delta\operatorname{softmax}$$ is zero, no failure
"""

# ╔═╡ 95bd6740-1a17-11eb-3372-21bac6614f53
function Δsoftmax(model, x::Vector, y)
	ŷ = model(x)
	return maximum(softmax(ŷ)) - softmax(ŷ)[argmax(y)]
end;

# ╔═╡ a9de0db0-1a2b-11eb-2144-e5025a1a9e5e
function Δsoftmax(model, x::Matrix, y)
	ŷ = model(x)
	softmaxŷ = softmax(ŷ)
	max_softmax = mapslices(maximum, softmaxŷ, dims=1)'
	true_yᵢ = mapslices(argmax, y, dims=1)
	true_softmax = [softmaxŷ[r,c] for (r,c) in zip(true_yᵢ, 1:size(y,2))]
	return vec(max_softmax - true_softmax)
end;

# ╔═╡ 3da097b0-2833-11eb-3566-11862e85da01
Δsoftmax(sut, X, Y)

# ╔═╡ 1f747b72-2834-11eb-0015-dd02035ce6bc
FailureRepresentation.Adversary.bce(0.5, 1) .+ log.(Δsoftmax(sut, X, Y) .+ eps(1.0))

# ╔═╡ 8fd9b780-2835-11eb-2698-ed54bb00d2c7
dl = Flux.Data.DataLoader([1, 1], [2, 2], [3, 3])

# ╔═╡ 974bb132-2835-11eb-29fa-d342c6b4712b
for (x,y) in dl
	@info x,y
end

# ╔═╡ 009237b0-1a15-11eb-3e6c-9ff29ea9a32a
begin
	idx = 1
	Plots.bar(0:9, Y[:,idx], xticks=0:9, fillalpha=0.6, label="true")
	Plots.bar!(0:9, softmax(sut(X[:,idx])), fillalpha=0.6, label="prediction")
end

# ╔═╡ e083cc00-1ac7-11eb-103a-13fb6fb3d586
Autoencoder.img(X[:,idx])'

# ╔═╡ 0e50baa0-1ad5-11eb-1603-cba64a9c41e7
Δsoftmax(sut, X[:, idx], Y[:, idx])

# ╔═╡ 99924160-1a35-11eb-0df0-ff718845bd4b
Xi, Yi = rand(testdata, 10)

# ╔═╡ 7b8cf5c0-1a30-11eb-149c-5fbf3a4d9a12
Δsoftmax(sut, Xi[:, 1], Yi[:, 1])

# ╔═╡ 769264b0-1a30-11eb-3742-7f8badcea17f
Δsoftmax(sut, Xi, Yi)

# ╔═╡ 9f53f2fe-1a35-11eb-241a-15d6dc131650
Δsoftmax(sut, Xi, Yi)

# ╔═╡ 2059e2d0-1ad5-11eb-21c5-abf90763e4b9
md"""
---
"""

# ╔═╡ 805fa7c0-1a15-11eb-2200-a166cafe9431
# function Δsoftmax_expanded(model, x, y)
# 	probs = softmax(model(x))
# 	ŷᵢₙₓ = onecold(probs)
# 	yᵢₙₓ = onecold(y)
# 	max_softmax = [probs[r,c] for (r,c) in zip(onecold(probs), 1:size(probs,2))]
# 	true_softmax = [probs[r,c] for (r,c) in zip(onecold(Yi), 1:size(probs,2))]
# 	return max_softmax - true_softmax
# end;

# ╔═╡ 4b5dc850-27c5-11eb-015b-a91f5e61a3cc
function confusion(testdata, model)
	confY = Int[]
	confY′ = Int[]
	for (x,y) in testdata
		push!(confY, onecold(y)...)
		push!(confY′, onecold(softmax(model(x)))...)
	end
	return confY, confY′
end

# ╔═╡ 5c81b510-27cf-11eb-2081-e7eb2f26ee09
function confusion_adversary(testdata, model, threshold=0.5)
	confY = Int[]
	confY′ = Int[]
	confY′_prob = Float64[]
	for (x,y) in testdata
		push!(confY, y...)
		push!(confY′, FailureRepresentation.Adversary.predict(model, x, threshold)...)
		push!(confY′_prob, model(x)...)
	end
	return confY, confY′, confY′_prob
end

# ╔═╡ 9038c9d0-2869-11eb-21e5-b5fe26e0435f
function confusion_random(testdata)
	confY = Int[]
	confY′ = Int[]
	for (x,y) in testdata
		push!(confY, y...)
		push!(confY′, rand() .>= 0.5)
	end
	return confY, confY′
end

# ╔═╡ 607c1fc0-27c5-11eb-2b0c-db7dd00ada2c
# confY, confY′ = confusion(x->sut(autoencoder(x)))
confY, confY′ = confusion(testdata, sut)

# ╔═╡ 41d1f9e0-27c6-11eb-22eb-cb7988e1c216
begin
	conf = confusmat(10, confY, confY′)
	# conf = conf ./ sum(conf, dims=2) # normalize per class
end

# ╔═╡ 345edf20-27c7-11eb-14c7-ed2a1f653528
correctrate(confY, confY′)

# ╔═╡ 74419690-27c8-11eb-2bef-95b760034b9f
FailureRepresentation.SystemUnderTest.accuracy(testdata, sut)

# ╔═╡ 87c13220-27c8-11eb-33a9-7b92300eb881
M = conf

# ╔═╡ 02e5174e-27c9-11eb-0f16-ffa235e1c87c
precision(M, i) = M[i,i] / sum(M[:,i])

# ╔═╡ 6e904330-27c9-11eb-1ba2-afae7afad48b
recall(M, i) = M[i,i] / sum(M[i,:])

# ╔═╡ 8369c970-27c9-11eb-202d-cbc34e5eb985
mean(precision(M, i) for i in 1:10), mean(recall(M, i) for i in 1:10)

# ╔═╡ 3bad2432-27c7-11eb-26bb-e51b66fa7698
errorrate(confY, confY′)

# ╔═╡ 2fffb5a0-27c5-11eb-28bc-35c520a05b32
begin
	close("all")
	figure(figsize=(4,3))
	Seaborn.heatmap(conf, annot=true, cmap="viridis",
		            xticklabels=0:9, yticklabels=0:9, fmt="d", 
		            annot_kws=Dict("size"=>8))
	title("Confusion matrix for system \$\\mathcal{S}\$")
	xlabel("predicted labels")
	ylabel("true labels")
	gcf()
end

# ╔═╡ adc62590-27da-11eb-1c83-c3065a5b1c54
PyPlot.savefig("confusion_sut.pgf", bbox_inches="tight")

# ╔═╡ c6a09110-27ce-11eb-196e-4365b178198a
adversary = BSON.load("../models/adversary_10.bson")[:adversary]

# ╔═╡ 906c4e70-27d0-11eb-16d3-15b462cf88e8
encoder = BSON.load("../models/encoder.bson")[:encoder]

# ╔═╡ fb4d27a0-27df-11eb-11dc-ddbfffd9fc84
decoder = BSON.load("../models/decoder.bson")[:decoder]

# ╔═╡ f5a06d50-27ce-11eb-17b4-c1599c1d93db
function convert_data(testdata, sut, encoder)
    x̃ = reduce(hcat, [encoder(x) for (x,_) in testdata])
    ỹ = reduce(vcat, [onecold(sut(x)) .!= onecold(y) for (x,y) in testdata])
    return DataLoader(x̃, ỹ)
end

# ╔═╡ e5c0b4d0-27ce-11eb-330b-2dfe47120ede
testdata_tilde = convert_data(testdata, sut, encoder)

# ╔═╡ 20b3f42e-27cf-11eb-00e7-73c83aa66e00
conf𝒜, conf𝒜′, conf𝒜′_prob = confusion_adversary(testdata_tilde, adversary)

# ╔═╡ 4bf84f10-27cf-11eb-0435-a12463478ae0
M𝒜 = confusmat(2, conf𝒜 .+ 1, conf𝒜′ .+ 1)

# ╔═╡ 15da21f0-27d0-11eb-025b-e7f9374b020b
begin
	close("all")
	PyPlot.figure(figsize=(4,3))
	Seaborn.heatmap(M𝒜, annot=true, cmap="viridis",
		            xticklabels=["not failure", "failure"],
		            yticklabels=["not failure", "failure"],
		            fmt="d", annot_kws=Dict("size"=>12))
	title("Confusion matrix for adversary \$\\mathcal{A}\$")
	xlabel("predicted failures")
	ylabel("true failures")
	gcf()
end

# ╔═╡ a1d0c69e-27da-11eb-163b-ab92a7859f62
PyPlot.savefig("confusion_adversary.pgf", bbox_inches="tight")

# ╔═╡ 2f42f450-27d0-11eb-1f41-cf1df27badc2
mean(precision(M𝒜, i) for i in 1:2), mean(recall(M𝒜, i) for i in 1:2)

# ╔═╡ bae39552-27d0-11eb-1c3b-cb0bfcacdb81
precision(M𝒜, 2), recall(M𝒜, 2)

# ╔═╡ 82a942e0-2869-11eb-17f2-5550a7d223a0
md"""
## Random baseline
"""

# ╔═╡ 867c6aee-2869-11eb-15bb-2b5934eaba9f
conf𝒜_r, conf𝒜′_r = confusion_random(testdata_tilde)

# ╔═╡ b8f05f50-2869-11eb-158b-47205e98fd88
M𝒜_r = confusmat(2, conf𝒜_r .+ 1, conf𝒜′_r .+ 1)

# ╔═╡ 65a19ee0-2869-11eb-0f9d-6d7759d79d9f
begin
	close("all")
	PyPlot.figure(figsize=(4,3))
	Seaborn.heatmap(M𝒜_r, annot=true, cmap="viridis",
		            xticklabels=["not failure", "failure"],
		            yticklabels=["not failure", "failure"],
		            fmt="d", annot_kws=Dict("size"=>8))
	title("Confusion matrix for adversary \$\\mathcal{A}\$")
	xlabel("predicted failures")
	ylabel("true failures")
	gcf()
end

# ╔═╡ 789ea780-286a-11eb-28dc-2185259857bd
precision(M𝒜_r, 2), recall(M𝒜_r, 2)

# ╔═╡ 798a4000-286a-11eb-1f12-e30338eb7ead
md"""
## ROC curve scikitlearn
"""

# ╔═╡ 268df680-27d4-11eb-175c-f79f746fbab7
@sk_import metrics: roc_curve

# ╔═╡ 5bc8ac00-27d4-11eb-31aa-edb430062b1b
@sk_import metrics: auc

# ╔═╡ 7e894f10-27d4-11eb-2b28-dbed5929b12b
begin
	rX, rY = [], []
	for (x,y) in testdata_tilde
		push!(rX, x...)
		push!(rY, y...)
	end
end

# ╔═╡ 96d3e030-27d4-11eb-39ea-db39558701a6
rY

# ╔═╡ 469ee2e2-27d4-11eb-3e68-39a1c8de84e3
fpr, tpr, _ = roc_curve(conf𝒜, conf𝒜′_prob)

# ╔═╡ 460879d0-27d5-11eb-2495-7920d55f677c
auc(fpr, tpr)

# ╔═╡ 15281992-27dc-11eb-2e1e-1b4d5028142b
md"""
## Threshold sweep
"""

# ╔═╡ 1985ce60-27dc-11eb-3aa6-f55d06695c2d
begin
	precision_t = []
	recall_t = []
	thresholds = 0:0.05:1
	for thresh in thresholds
		conf𝒜_t, conf𝒜′_t, conf𝒜′_prob_t =
			confusion_adversary(testdata_tilde, adversary, thresh)
		M𝒜_t = confusmat(2, conf𝒜_t .+ 1, conf𝒜′_t .+ 1)
		push!(precision_t, precision(M𝒜_t, 2))
		push!(recall_t, recall(M𝒜_t, 2))
	end
end

# ╔═╡ e4bf4200-27dc-11eb-051f-8b50f7f4bdd8
begin
	close("all")
	figure(figsize=(4,3))
	PyPlot.plot(thresholds, precision_t, color="blue", marker=".")
	PyPlot.plot(thresholds, recall_t, color="red", marker=".")
	legend(["Precision", "Recall"])
	title("Prediction threshold sweep: precision and recall")
	xlabel("prediction threshold: \$\\hat y \\ge {threshold}\$")
	ylabel("precision and recall")
	PyPlot.plot([0.5005, 0.5005], [-0.01, recall_t[11]], linestyle=":", color="gray")
	ylim([-0.01, 1.05])
	gcf()
end

# ╔═╡ ae24ea8e-27de-11eb-216e-55bf2ed07117
PyPlot.savefig("threshold_sweep.pgf", bbox_inches="tight")

# ╔═╡ 1cad742e-27dc-11eb-1d62-ed5ef14f01a0
md"""
## ROC curves
"""

# ╔═╡ 430dc580-27d7-11eb-2edd-81e1104ab57b
ColorSchemes.Blues_9

# ╔═╡ de03e060-27d7-11eb-19e4-f303bb9078f3
rgb(c) = (red(c), green(c), blue(c))

# ╔═╡ edbb0240-27d7-11eb-1c26-e3a457165160
rgb(ColorSchemes.Reds_9[1])

# ╔═╡ 384e63a0-27d9-11eb-1540-7731666f9aac
# using PGFPlots

# ╔═╡ dde1c780-27d9-11eb-2345-b9067ba467b7
begin
	matplotlib.rc("font", family=["serif"]) # sans-serif keyword not supported
	# matplotlib.rc("font", serif=["Helvetica"]) # matplotlib.rc("font", serif=["Palatino"])
	matplotlib.rc("text", usetex=true)
	matplotlib.rc("pgf", rcfonts=false)
end

# ╔═╡ c0cf6762-27d4-11eb-3a68-e1fc72404bd7
begin
	close("all")
	figure(figsize=(4,3))
	legends = []
	for i in 1:10
		color = i >= 10 ? "black" : rgb(ColorSchemes.Blues_9[i])
		adversary_i = BSON.load("../models/adversary_$i.bson")[:adversary]
		conf𝒜_i, conf𝒜′_i, conf𝒜′_prob_i =
			confusion_adversary(testdata_tilde, adversary_i)
		fprᵢ, tprᵢ, _ = roc_curve(conf𝒜_i, conf𝒜′_prob_i)
		PyPlot.plot(fprᵢ, tprᵢ, color=color)
		push!(legends, "AUC\$_{$i}\$ = \$$(round(auc(fprᵢ, tprᵢ), digits=3))\$")
	end
	ylabel("true positive rate")
	xlabel("false positive rate")
	title("Receiver operating characteristic (ROC) curve per iteration \$t \\in T\$")
	legend(legends, loc="center left", bbox_to_anchor=(1.04, 0.5))
	PyPlot.plot(0:1, 0:1, linestyle="--", color="gray")
	gcf()
end

# ╔═╡ bc9ee040-27d8-11eb-2ae2-fd3ad839dffb
PyPlot.savefig("roc.pgf", bbox_inches="tight")

# ╔═╡ 880aede0-27df-11eb-342c-af78db5ae8bb
md"""
## Qualitative analysis
Most likely failure via the prediction, but wasn't a failure.
"""

# ╔═╡ 9207b11e-27df-11eb-262b-83039547db91
arg = argmax([maximum(adversary(x)) for (x,y) in testdata_tilde])

# ╔═╡ c06496a0-27df-11eb-0557-35927114712c
arg_x, arg_y = collect(testdata_tilde)[arg]

# ╔═╡ e9f2b600-27df-11eb-3190-d3a359b8b5cb
Autoencoder.img(decoder(vec(arg_x)))'

# ╔═╡ 314cdad0-27e0-11eb-261b-359761d4268b
Flux.flatten(collect(testdata))

# ╔═╡ 1a62c410-27e0-11eb-14ab-339edb9da195
function select_top_k(model, testdata, k=5, rev=true)
    return partialsortperm([model(x)[1] for (x,y) in testdata], 1:k, rev=rev)
end

# ╔═╡ b41442a0-2830-11eb-03d0-ed6c96e9dde9
function select_bottom_k_false_negatives(model, testdata, k=5, rev=false)
	idx = []
	predicted_value = []
	for (i,(x,y)) in enumerate(testdata)
		# if false negative
		if y[1] && !FailureRepresentation.Adversary.predict(model, x)[1]
			push!(idx, i)
			push!(predicted_value, model(x)[1])
		end
	end
	perm = partialsortperm(predicted_value, 1:k, rev=rev)
	return idx[perm], predicted_value[perm]			
end

# ╔═╡ 2117c860-287a-11eb-11f2-5304606bcdf3
function select_top_k_true_positives(model, testdata, k=5, rev=true)
	idx = []
	predicted_value = []
	for (i,(x,y)) in enumerate(testdata)
		# if true positive
		if y[1] && FailureRepresentation.Adversary.predict(model, x)[1]
			push!(idx, i)
			push!(predicted_value, model(x)[1])
		end
	end
	perm = partialsortperm(predicted_value, 1:k, rev=rev)
	return idx[perm], predicted_value[perm]			
end

# ╔═╡ ab2aeb90-287a-11eb-0a90-ff7e62c0f950
md"""
## False negatives
"""

# ╔═╡ 4b99eee0-2831-11eb-20e4-cbae0c67e4fb
bot_false_negs = select_bottom_k_false_negatives(adversary, testdata_tilde, 10)

# ╔═╡ ce96ccf0-2831-11eb-3f91-4bbe163284b4
false_neg_decoded_images = Autoencoder.img.([decoder(vec(args_x)) for (args_x, args_y) in collect(testdata_tilde)[bot_false_negs[1]]])'

# ╔═╡ 4499ad50-2832-11eb-1316-a3370fe12816
false_neg_true_failures = [args_y[1] for (args_x, args_y) in collect(testdata_tilde)[bot_false_negs[1]]]

# ╔═╡ 16d1a010-287a-11eb-1ef2-eb3e698fe85f
md"""
## True positives
"""

# ╔═╡ 33caec30-287a-11eb-099b-5b6571982daf
top_true_pos = select_top_k_true_positives(adversary, testdata_tilde, 10)

# ╔═╡ 4af4cb10-287a-11eb-3718-f7d59a2c0eed
true_pos_decoded_images = Autoencoder.img.([decoder(vec(args_x)) for (args_x, args_y) in collect(testdata_tilde)[top_true_pos[1]]])'

# ╔═╡ cdf44130-287a-11eb-13dd-796a7a81dd8a
md"""
## Likely failures
"""

# ╔═╡ e02a6c70-2830-11eb-08d9-631b8f5c8166
testdata_tilde

# ╔═╡ 9c05b770-27e0-11eb-2116-55c62bb8e032
args = select_top_k(adversary, testdata_tilde, 10)

# ╔═╡ 89daa100-2830-11eb-0ab0-09728faf861b
args_bot = select_top_k(adversary, testdata_tilde, 10, false)

# ╔═╡ 9ab0ee80-2830-11eb-15eb-6dcee557eadd
[minimum(adversary(x)) for (x,y) in testdata_tilde][args_bot]

# ╔═╡ a4de85c0-2830-11eb-181a-079c0c560cb6
true_failures_bot = [args_y[1] for (args_x, args_y) in collect(testdata_tilde)[args_bot]]

# ╔═╡ b4aabd40-27e3-11eb-1d99-4bb53010e540
[maximum(adversary(x)) for (x,y) in testdata_tilde][args]

# ╔═╡ b31de312-27e0-11eb-0a4e-cde578822d68
decoded_images = Autoencoder.img.([decoder(vec(args_x)) for (args_x, args_y) in collect(testdata_tilde)[args]])'

# ╔═╡ df09c890-27e0-11eb-1509-afdf248cd717
true_failures = [args_y[1] for (args_x, args_y) in collect(testdata_tilde)[args]]

# ╔═╡ 7a323e12-27e1-11eb-3720-a35f5fff7948
function flatten_testdata(testdata)
	X, Y = [], []
	for (x,y) in testdata
		@show size(x), size(y)
		for i in 1:size(y,2)
			push!(X, x[:,i])
			push!(Y, y[:,i])
		end
	end
	@info size(X), size(Y)
	return Flux.Data.DataLoader(X, Y)
end

# ╔═╡ 90073b4e-27e1-11eb-1b41-9d02c393df7c
ftestdata = flatten_testdata(testdata)

# ╔═╡ f7aada50-2831-11eb-2318-0f64039d365d
false_neg_true_inputs = [x[1] for x in first.(collect(ftestdata))[bot_false_negs[1]]]

# ╔═╡ eb076110-2831-11eb-2151-6ba1beee687c
false_neg_true_images = Autoencoder.img.([x for x in false_neg_true_inputs])'

# ╔═╡ 83009a90-287d-11eb-364a-014efae85a3a
fn_true_labels = [onecold(y[1]) - 1 for y in last.(collect(ftestdata))[bot_false_negs[1]]]

# ╔═╡ 89381872-287d-11eb-0991-bdd241001a4c
fn_sut_labels = [onecold(sut(x[1])) - 1 for x in first.(collect(ftestdata))[bot_false_negs[1]]]

# ╔═╡ 5e6a0660-287a-11eb-0308-874829b878aa
true_pos_true_inputs = [x[1] for x in first.(collect(ftestdata))[top_true_pos[1]]]

# ╔═╡ 678d3280-287a-11eb-2c43-bd919979d395
true_pos_true_images = Autoencoder.img.([x for x in true_pos_true_inputs])'

# ╔═╡ 675fb282-287d-11eb-0c1d-7d21c31b3ae6
tp_true_labels = [onecold(y[1]) - 1 for y in last.(collect(ftestdata))[top_true_pos[1]]]

# ╔═╡ 72e42730-287d-11eb-26d4-e9cbe9b7b2cc
tp_sut_labels = [onecold(sut(x[1])) - 1 for x in first.(collect(ftestdata))[top_true_pos[1]]]

# ╔═╡ f18d3460-27e1-11eb-0351-f9b44125b4bc
true_labels = [onecold(y[1]) - 1 for y in last.(collect(ftestdata))[args]]

# ╔═╡ 885a17f0-27e2-11eb-2720-1d13935ac861
sut_labels = [onecold(sut(x[1])) - 1 for x in first.(collect(ftestdata))[args]]

# ╔═╡ 563506e0-27e2-11eb-154d-cb3af1504823
true_inputs = [x[1] for x in first.(collect(ftestdata))[args]]

# ╔═╡ 502164b0-27e2-11eb-1201-b93ded002d96
true_images = Autoencoder.img.([x for x in true_inputs])'

# ╔═╡ 90ae7b00-27e5-11eb-0991-532c3e8e3941
redimg(x::Vector) = reshape([RGB(clamp(i, 0, 1), 0, 0) for i in x], 28, 28)

# ╔═╡ f6f1ae00-27e5-11eb-2538-6dcf631a43b3
greenimg(x::Vector) = reshape([RGB(0, clamp(i, 0, 1), 0) for i in x], 28, 28)

# ╔═╡ 34b11b30-2832-11eb-0cb5-775f817a2d7c
false_neg_failure_indicators = [y ? greenimg(fill(0.4, 28^2)) : redimg(fill(0.4, 28^2)) for y in false_neg_true_failures]'

# ╔═╡ 291ceb80-27e4-11eb-32d3-69542e48e4e7
failure_indicators = [y ? greenimg(fill(0.4, 28^2)) : redimg(fill(0.4, 28^2)) for y in true_failures]'

# ╔═╡ a51cbc50-27e5-11eb-11d0-317e3a30e5ea
redimg(fill(0.4, 28^2)), greenimg(fill(0.4, 28^2))

# ╔═╡ 8c6f5540-27e6-11eb-226b-3d61418cc764
hcat([Gray.(reshape(args_x, 8, 8)) for (args_x, args_y) in collect(testdata_tilde)[args]]...)

# ╔═╡ a8bb0e30-27e4-11eb-1578-53806dd0f0e9
Autoencoder.img(decoder(randn(64)))'

# ╔═╡ 32323f80-27e5-11eb-1581-3537f89b6549
hcat((true_images .- decoded_images)...)

# ╔═╡ 54351b60-27e6-11eb-3898-15504bbff380
Gray.(reshape(collect(testdata_tilde)[6][1], 8, 8))

# ╔═╡ 5c03be90-27e7-11eb-0d07-6529c6633b4f
function onehotvector(i, n, v=1)
	X = zeros(n)
	X[i] = v
	return X
end

# ╔═╡ 0a565d00-2832-11eb-3110-a9c7d74cb153
false_neg_feature_images = Autoencoder.img.([decoder(onehotvector(argmax(softmax(vec(args_x))), 64, 10)) for (args_x, args_y) in collect(testdata_tilde)[bot_false_negs[1]]])'

# ╔═╡ 15abd26e-2832-11eb-1cae-818f96fce3f9
false_negatives = hcat(vcat.(false_neg_true_images, false_neg_feature_images, false_neg_decoded_images)...)

# ╔═╡ 70830d10-287a-11eb-269c-c3fa1eaeb9e2
true_pos_feature_images = Autoencoder.img.([decoder(onehotvector(argmax(softmax(vec(args_x))), 64, 10)) for (args_x, args_y) in collect(testdata_tilde)[top_true_pos[1]]])'

# ╔═╡ 05517e02-287a-11eb-167c-0935bd9a2a70
true_positives = hcat(vcat.(true_pos_true_images, true_pos_feature_images, true_pos_decoded_images)...)

# ╔═╡ 4999e350-27e8-11eb-143b-dd46d2a2a8d9
feature_images = Autoencoder.img.([decoder(onehotvector(argmax(softmax(vec(args_x))), 64, 10)) for (args_x, args_y) in collect(testdata_tilde)[args]])'

# ╔═╡ eddba8a0-27e2-11eb-3c2b-456abfb6d91c
likely_failures = hcat(vcat.(true_images, feature_images, decoded_images, failure_indicators)...)

# ╔═╡ dcb73e50-27e6-11eb-3f42-35c9f67c96da
Autoencoder.img(decoder(onehotvector(1,64,10)))'

# ╔═╡ 9bd64200-27eb-11eb-2c81-f7e1bb552e84
function twohotvector(i, j, n, v=1)
	X = zeros(n)
	X[i] = v
	X[j] = v
	return X
end

# ╔═╡ 7a474bb2-27ec-11eb-3a1b-4d6f11cc8042
function threehotvector(i, j, k, n, v=1)
	X = zeros(n)
	X[i] = v
	X[j] = v
	X[k] = v
	return X
end

# ╔═╡ 6ff44460-27e7-11eb-371e-9dddb1d9a76c
onehotvector(1,10)

# ╔═╡ 8adfbf20-27e7-11eb-0cec-91f326f42e55
ex = onehotvector(1,64)

# ╔═╡ 0ac1d330-27e9-11eb-148e-156e7d670757
hcat(vcat.([[Autoencoder.img(decoder(onehotvector(i, 64, 10)))' for i in (8j - 7):8j] for j in 1:8]...)...)

# ╔═╡ 734e20f0-27eb-11eb-11fb-d51498c1592f
Autoencoder.img(decoder(twohotvector(1, 6, 64, 10)))'

# ╔═╡ c1f48f00-27eb-11eb-13af-599491b28241
argmax([[adversary(twohotvector(i, j, 64, 10)) for i in 1:64] for j in 1:64])

# ╔═╡ 0bd585c0-27ec-11eb-32ea-f33ef2084890
combinations = vec([(i,j) for i in 1:64, j in 1:64])

# ╔═╡ 81a56f40-27ec-11eb-1bf8-b30e0f259dc4
combinations3 = vec([(i,j,k) for i in 1:64, j in 1:64, k in 1:64])

# ╔═╡ 2e574b10-27ec-11eb-26ea-db2bfc5278fe
max2hot = argmax([adversary(twohotvector(i, j, 64, 10)) for (i,j) in combinations])

# ╔═╡ 67663a10-27ec-11eb-0173-27217c95d913
adversary(twohotvector(combinations[max2hot]..., 64, 1))

# ╔═╡ eb2cc630-27eb-11eb-0496-831311ac09e9
Autoencoder.img(decoder(twohotvector(combinations[max2hot]..., 64, 10)))'

# ╔═╡ 8a935a92-27ec-11eb-3508-d9ca334ea4b5
max3hot = argmax([adversary(threehotvector(i, j, k, 64, 10)) for (i,j,k) in combinations3])

# ╔═╡ 96b791b0-27ec-11eb-27a4-bf55cb102ee2
Autoencoder.img(decoder(threehotvector(combinations3[max3hot]..., 64, 10)))'

# ╔═╡ 292d3070-27ea-11eb-088b-efc17f03f787
maximum(adversary(onehotvector(i, 64, 1)) for i in 1:64)

# ╔═╡ 273869e0-282d-11eb-12ae-e90ddca72641
@bind intensity Slider(1:100, default=10, show_value=true)

# ╔═╡ 47fee610-27ea-11eb-0a2c-c1b0c93f4446
encoding_map = hcat(vcat.([[Autoencoder.img(decoder(onehotvector(i, 64, intensity*adversary(onehotvector(i,64))[1])))' ./ intensity for i in (8j - 7):8j] for j in 1:8]...)...)

# ╔═╡ 29974c00-282e-11eb-044e-a972ed21be77
begin
	close("all")
	figure(figsize=(4,3))
	ticks8 = range(28/2, stop=8*28-(28/2), length=8)
	Seaborn.heatmap(convert(Array{Float32, 2}, encoding_map),
		            cmap="viridis", xticklabels=1:8, yticklabels=1:8)
	xticks(ticks8)
	yticks(ticks8)
	title("Likelihood sweep over low-dimensional features")
	xlabel("feature index \$i\$")
	ylabel("feature index \$j\$")
	gcf()
end

# ╔═╡ 0e986a50-2870-11eb-2bdc-ef8bd3938307
PyPlot.savefig("feature-likelihood.pgf", bbox_inches="tight")

# ╔═╡ 5936b910-282d-11eb-107f-69a8742adfa0
@bind likelihood_intensity Slider(1:1)

# ╔═╡ 632eb3a0-2828-11eb-1933-7b74bf88bcdc
begin
	close("all")
	figure(figsize=(4,3))
	Seaborn.heatmap(reshape([adversary(encoder(onehotvector(i, 28^2, likelihood_intensity)))[1] for i in 1:28^2], 28, 28)', cmap="viridis", xticklabels=0:7:28, yticklabels=0:7:28)
	title("Failure likelihood per pixel")
	xticks(0:7:28)
	yticks(0:7:28)
	xlabel("\$x\$-pixel")
	ylabel("\$y\$-label")
	gcf()
end

# ╔═╡ 3584b5b0-2870-11eb-3d47-7762282618a9
PyPlot.savefig("failure-likelihood.pgf", bbox_inches="tight")

# ╔═╡ ab1efe50-2872-11eb-0a14-f900c5907e05
md"""
## SUT metrics
"""

# ╔═╡ Cell order:
# ╠═4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
# ╠═5c706e30-19b1-11eb-18ad-5190eca820b1
# ╠═330f1d40-282d-11eb-29e0-f3affbba2711
# ╠═54164520-19b6-11eb-376c-b5246d05c3a6
# ╠═30bd5680-19b2-11eb-3ed2-81593376e639
# ╠═1732d5e0-19b3-11eb-3b9d-8b5b61fc38ae
# ╠═1e4525a0-27a4-11eb-0422-9d0d7dc3dd9c
# ╠═24ef3930-19b3-11eb-0d9d-e3a0733ba1da
# ╠═c3996450-1a14-11eb-2b50-934336a06d5a
# ╠═b3418cb2-19b3-11eb-25b4-4b48a12ad82e
# ╠═c2f9dec0-27a3-11eb-0b14-75176c151264
# ╠═8becbb82-19b3-11eb-2c35-b17e8dee1ed2
# ╠═708c22b0-19b1-11eb-3ee7-f32bf48ebf1f
# ╠═1cef6c80-19b5-11eb-1f69-7356f410b1fc
# ╠═3cf13db0-19b5-11eb-081b-ed8141b6b8d7
# ╠═f4fbfc40-19b2-11eb-0ee1-0b3e9ece6008
# ╠═8cde0240-1a14-11eb-0274-090f1dcdeb0d
# ╠═d4fcfdae-1a14-11eb-0a76-5dfe76d1754a
# ╟─b6901270-1a16-11eb-09a2-c1073bff3f8f
# ╟─5d045420-2833-11eb-1f67-59da9ee8ef92
# ╠═95bd6740-1a17-11eb-3372-21bac6614f53
# ╠═7b8cf5c0-1a30-11eb-149c-5fbf3a4d9a12
# ╠═a9de0db0-1a2b-11eb-2144-e5025a1a9e5e
# ╠═769264b0-1a30-11eb-3742-7f8badcea17f
# ╠═3da097b0-2833-11eb-3566-11862e85da01
# ╠═1f747b72-2834-11eb-0015-dd02035ce6bc
# ╠═8fd9b780-2835-11eb-2698-ed54bb00d2c7
# ╠═974bb132-2835-11eb-29fa-d342c6b4712b
# ╠═009237b0-1a15-11eb-3e6c-9ff29ea9a32a
# ╠═e083cc00-1ac7-11eb-103a-13fb6fb3d586
# ╠═0e50baa0-1ad5-11eb-1603-cba64a9c41e7
# ╠═99924160-1a35-11eb-0df0-ff718845bd4b
# ╠═9f53f2fe-1a35-11eb-241a-15d6dc131650
# ╟─2059e2d0-1ad5-11eb-21c5-abf90763e4b9
# ╠═805fa7c0-1a15-11eb-2200-a166cafe9431
# ╠═2c1e5cc0-27c5-11eb-1e52-6121d656b973
# ╠═4b5dc850-27c5-11eb-015b-a91f5e61a3cc
# ╠═5c81b510-27cf-11eb-2081-e7eb2f26ee09
# ╠═9038c9d0-2869-11eb-21e5-b5fe26e0435f
# ╠═607c1fc0-27c5-11eb-2b0c-db7dd00ada2c
# ╠═27c76940-27c6-11eb-0ba2-c9213d619fb4
# ╠═41d1f9e0-27c6-11eb-22eb-cb7988e1c216
# ╠═345edf20-27c7-11eb-14c7-ed2a1f653528
# ╠═74419690-27c8-11eb-2bef-95b760034b9f
# ╠═87c13220-27c8-11eb-33a9-7b92300eb881
# ╠═02e5174e-27c9-11eb-0f16-ffa235e1c87c
# ╠═6e904330-27c9-11eb-1ba2-afae7afad48b
# ╠═8369c970-27c9-11eb-202d-cbc34e5eb985
# ╠═3bad2432-27c7-11eb-26bb-e51b66fa7698
# ╠═2fffb5a0-27c5-11eb-28bc-35c520a05b32
# ╠═adc62590-27da-11eb-1c83-c3065a5b1c54
# ╠═c6a09110-27ce-11eb-196e-4365b178198a
# ╠═906c4e70-27d0-11eb-16d3-15b462cf88e8
# ╠═fb4d27a0-27df-11eb-11dc-ddbfffd9fc84
# ╠═e5c0b4d0-27ce-11eb-330b-2dfe47120ede
# ╠═f5a06d50-27ce-11eb-17b4-c1599c1d93db
# ╠═20b3f42e-27cf-11eb-00e7-73c83aa66e00
# ╠═4bf84f10-27cf-11eb-0435-a12463478ae0
# ╠═15da21f0-27d0-11eb-025b-e7f9374b020b
# ╠═a1d0c69e-27da-11eb-163b-ab92a7859f62
# ╠═2f42f450-27d0-11eb-1f41-cf1df27badc2
# ╠═bae39552-27d0-11eb-1c3b-cb0bfcacdb81
# ╠═82a942e0-2869-11eb-17f2-5550a7d223a0
# ╠═867c6aee-2869-11eb-15bb-2b5934eaba9f
# ╠═b8f05f50-2869-11eb-158b-47205e98fd88
# ╠═65a19ee0-2869-11eb-0f9d-6d7759d79d9f
# ╠═789ea780-286a-11eb-28dc-2185259857bd
# ╟─798a4000-286a-11eb-1f12-e30338eb7ead
# ╠═5455e46e-27d3-11eb-1478-0f9879fc2215
# ╠═268df680-27d4-11eb-175c-f79f746fbab7
# ╠═5bc8ac00-27d4-11eb-31aa-edb430062b1b
# ╠═7e894f10-27d4-11eb-2b28-dbed5929b12b
# ╠═96d3e030-27d4-11eb-39ea-db39558701a6
# ╠═469ee2e2-27d4-11eb-3e68-39a1c8de84e3
# ╠═460879d0-27d5-11eb-2495-7920d55f677c
# ╟─15281992-27dc-11eb-2e1e-1b4d5028142b
# ╠═1985ce60-27dc-11eb-3aa6-f55d06695c2d
# ╠═e4bf4200-27dc-11eb-051f-8b50f7f4bdd8
# ╠═ae24ea8e-27de-11eb-216e-55bf2ed07117
# ╟─1cad742e-27dc-11eb-1d62-ed5ef14f01a0
# ╠═4147f9a0-27d7-11eb-32de-a9d04fc45211
# ╠═430dc580-27d7-11eb-2edd-81e1104ab57b
# ╠═de03e060-27d7-11eb-19e4-f303bb9078f3
# ╠═edbb0240-27d7-11eb-1c26-e3a457165160
# ╠═384e63a0-27d9-11eb-1540-7731666f9aac
# ╠═dde1c780-27d9-11eb-2345-b9067ba467b7
# ╠═c0cf6762-27d4-11eb-3a68-e1fc72404bd7
# ╠═bc9ee040-27d8-11eb-2ae2-fd3ad839dffb
# ╟─880aede0-27df-11eb-342c-af78db5ae8bb
# ╠═9207b11e-27df-11eb-262b-83039547db91
# ╠═c06496a0-27df-11eb-0557-35927114712c
# ╠═e9f2b600-27df-11eb-3190-d3a359b8b5cb
# ╠═314cdad0-27e0-11eb-261b-359761d4268b
# ╠═1a62c410-27e0-11eb-14ab-339edb9da195
# ╠═b41442a0-2830-11eb-03d0-ed6c96e9dde9
# ╠═2117c860-287a-11eb-11f2-5304606bcdf3
# ╟─ab2aeb90-287a-11eb-0a90-ff7e62c0f950
# ╠═4b99eee0-2831-11eb-20e4-cbae0c67e4fb
# ╠═ce96ccf0-2831-11eb-3f91-4bbe163284b4
# ╠═f7aada50-2831-11eb-2318-0f64039d365d
# ╠═eb076110-2831-11eb-2151-6ba1beee687c
# ╠═0a565d00-2832-11eb-3110-a9c7d74cb153
# ╠═4499ad50-2832-11eb-1316-a3370fe12816
# ╠═34b11b30-2832-11eb-0cb5-775f817a2d7c
# ╠═15abd26e-2832-11eb-1cae-818f96fce3f9
# ╠═83009a90-287d-11eb-364a-014efae85a3a
# ╠═89381872-287d-11eb-0991-bdd241001a4c
# ╟─16d1a010-287a-11eb-1ef2-eb3e698fe85f
# ╠═33caec30-287a-11eb-099b-5b6571982daf
# ╠═4af4cb10-287a-11eb-3718-f7d59a2c0eed
# ╠═5e6a0660-287a-11eb-0308-874829b878aa
# ╠═678d3280-287a-11eb-2c43-bd919979d395
# ╠═70830d10-287a-11eb-269c-c3fa1eaeb9e2
# ╠═05517e02-287a-11eb-167c-0935bd9a2a70
# ╠═675fb282-287d-11eb-0c1d-7d21c31b3ae6
# ╠═72e42730-287d-11eb-26d4-e9cbe9b7b2cc
# ╟─cdf44130-287a-11eb-13dd-796a7a81dd8a
# ╠═e02a6c70-2830-11eb-08d9-631b8f5c8166
# ╠═9c05b770-27e0-11eb-2116-55c62bb8e032
# ╠═89daa100-2830-11eb-0ab0-09728faf861b
# ╠═9ab0ee80-2830-11eb-15eb-6dcee557eadd
# ╠═a4de85c0-2830-11eb-181a-079c0c560cb6
# ╠═b4aabd40-27e3-11eb-1d99-4bb53010e540
# ╠═b31de312-27e0-11eb-0a4e-cde578822d68
# ╠═4999e350-27e8-11eb-143b-dd46d2a2a8d9
# ╠═df09c890-27e0-11eb-1509-afdf248cd717
# ╠═7a323e12-27e1-11eb-3720-a35f5fff7948
# ╠═90073b4e-27e1-11eb-1b41-9d02c393df7c
# ╠═f18d3460-27e1-11eb-0351-f9b44125b4bc
# ╠═885a17f0-27e2-11eb-2720-1d13935ac861
# ╠═563506e0-27e2-11eb-154d-cb3af1504823
# ╠═502164b0-27e2-11eb-1201-b93ded002d96
# ╠═291ceb80-27e4-11eb-32d3-69542e48e4e7
# ╠═90ae7b00-27e5-11eb-0991-532c3e8e3941
# ╠═f6f1ae00-27e5-11eb-2538-6dcf631a43b3
# ╠═a51cbc50-27e5-11eb-11d0-317e3a30e5ea
# ╠═eddba8a0-27e2-11eb-3c2b-456abfb6d91c
# ╠═8c6f5540-27e6-11eb-226b-3d61418cc764
# ╠═a8bb0e30-27e4-11eb-1578-53806dd0f0e9
# ╠═32323f80-27e5-11eb-1581-3537f89b6549
# ╠═54351b60-27e6-11eb-3898-15504bbff380
# ╠═dcb73e50-27e6-11eb-3f42-35c9f67c96da
# ╠═5c03be90-27e7-11eb-0d07-6529c6633b4f
# ╠═9bd64200-27eb-11eb-2c81-f7e1bb552e84
# ╠═7a474bb2-27ec-11eb-3a1b-4d6f11cc8042
# ╠═6ff44460-27e7-11eb-371e-9dddb1d9a76c
# ╠═8adfbf20-27e7-11eb-0cec-91f326f42e55
# ╠═0ac1d330-27e9-11eb-148e-156e7d670757
# ╠═734e20f0-27eb-11eb-11fb-d51498c1592f
# ╠═c1f48f00-27eb-11eb-13af-599491b28241
# ╠═0bd585c0-27ec-11eb-32ea-f33ef2084890
# ╠═81a56f40-27ec-11eb-1bf8-b30e0f259dc4
# ╠═2e574b10-27ec-11eb-26ea-db2bfc5278fe
# ╠═67663a10-27ec-11eb-0173-27217c95d913
# ╠═eb2cc630-27eb-11eb-0496-831311ac09e9
# ╠═8a935a92-27ec-11eb-3508-d9ca334ea4b5
# ╠═96b791b0-27ec-11eb-27a4-bf55cb102ee2
# ╠═292d3070-27ea-11eb-088b-efc17f03f787
# ╠═273869e0-282d-11eb-12ae-e90ddca72641
# ╠═47fee610-27ea-11eb-0a2c-c1b0c93f4446
# ╠═29974c00-282e-11eb-044e-a972ed21be77
# ╠═0e986a50-2870-11eb-2bdc-ef8bd3938307
# ╠═5936b910-282d-11eb-107f-69a8742adfa0
# ╠═632eb3a0-2828-11eb-1933-7b74bf88bcdc
# ╠═3584b5b0-2870-11eb-3d47-7762282618a9
# ╠═ab1efe50-2872-11eb-0a14-f900c5907e05
