### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

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
Δsoftmax(sut, Xi[:, 6], Yi[:, 6])

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
close("all"); Seaborn.heatmap(conf, annot=true, cmap="viridis", xticklabels=0:9, yticklabels=0:9, fmt="d", annot_kws=Dict("size"=>8)); title("Confusion matrix for system \$\\mathcal{S}\$"); xlabel("predicted labels"); ylabel("true labels"); gcf()

# ╔═╡ adc62590-27da-11eb-1c83-c3065a5b1c54
PyPlot.savefig("confusion_sut.pgf")

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
conf𝒜, conf𝒜′, conf𝒜′_prob = confusion_adversary(testdata_tilde, adversary, 0.6)

# ╔═╡ 4bf84f10-27cf-11eb-0435-a12463478ae0
M𝒜 = confusmat(2, conf𝒜 .+ 1, conf𝒜′ .+ 1)

# ╔═╡ 15da21f0-27d0-11eb-025b-e7f9374b020b
begin
	close("all")
	Seaborn.heatmap(M𝒜, annot=true, cmap="viridis",
		            xticklabels=["not failure", "failure"],
		            yticklabels=["not failure", "failure"],
		            fmt="d", annot_kws=Dict("size"=>8))
	title("Confusion matrix for adversary \$\\mathcal{A}\$")
	xlabel("predicted failures")
	ylabel("true failures")
	gcf()
end

# ╔═╡ a1d0c69e-27da-11eb-163b-ab92a7859f62
PyPlot.savefig("confusion_adversary.pgf")

# ╔═╡ 2f42f450-27d0-11eb-1f41-cf1df27badc2
mean(precision(M𝒜, i) for i in 1:2), mean(recall(M𝒜, i) for i in 1:2)

# ╔═╡ bae39552-27d0-11eb-1c3b-cb0bfcacdb81
precision(M𝒜, 2), recall(M𝒜, 2)

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
	PyPlot.plot(thresholds, precision_t, color="blue", marker=".")
	PyPlot.plot(thresholds, recall_t, color="red", marker=".")
	legend(["Precision", "Recall"])
	title("Prediction threshold sweep: precision and recall")
	xlabel("prediction threshold: ") # \$\\hat y \\ge \\rm{threshold}\$
	ylabel("precision and recall")
	PyPlot.plot([0.5005, 0.5005], [-0.01, recall_t[11]], linestyle=":", color="gray")
	ylim([-0.01, 1.05])
	gcf()
end

# ╔═╡ ae24ea8e-27de-11eb-216e-55bf2ed07117
PyPlot.savefig("threshold_sweep.pgf")

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
	legends = []
	for i in 1:10
		color = i == 10 ? "black" : rgb(ColorSchemes.Blues_9[i])
		adversary_i = BSON.load("../models/adversary_$i.bson")[:adversary]
		conf𝒜_i, conf𝒜′_i, conf𝒜′_prob_i =
			confusion_adversary(testdata_tilde, adversary_i)
		fprᵢ, tprᵢ, _ = roc_curve(conf𝒜_i, conf𝒜′_prob_i)
		PyPlot.plot(fprᵢ, tprᵢ, color=color)
		push!(legends, "AUC\$_{$i}\$ = \$$(round(auc(fprᵢ, tprᵢ), digits=3))\$")
	end
	ylabel("true positive rate")
	xlabel("false positive rate")
	title("Receiver operating characteristic (ROC) curve per iteration \$t\$")
	legend(legends)
	PyPlot.plot(0:1, 0:1, linestyle="--", color="gray")
	gcf()
end

# ╔═╡ bc9ee040-27d8-11eb-2ae2-fd3ad839dffb
PyPlot.savefig("roc.pgf")

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
function select_top_k(model, testdata, k=5)
    return partialsortperm([model(x)[1] for (x,y) in testdata], 1:k, rev=true)
end

# ╔═╡ 9c05b770-27e0-11eb-2116-55c62bb8e032
args = select_top_k(adversary, testdata_tilde, 10)

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

# ╔═╡ 291ceb80-27e4-11eb-32d3-69542e48e4e7
failure_indicators = [y ? greenimg(fill(0.4, 28^2)) : redimg(fill(0.4, 28^2)) for y in true_failures]'

# ╔═╡ a51cbc50-27e5-11eb-11d0-317e3a30e5ea
redimg(fill(0.4, 28^2))

# ╔═╡ fe084142-27e5-11eb-278e-d18844c41964
greenimg(fill(0.4, 28^2))

# ╔═╡ 8c6f5540-27e6-11eb-226b-3d61418cc764
hcat([Gray.(reshape(args_x, 8, 8)) for (args_x, args_y) in collect(testdata_tilde)[args]]...)

# ╔═╡ a9910dc0-27e2-11eb-1b6c-bf7fbc836b6c
# function sample_failures(adversary, testdata_tilde, k=5)
# 	args = select_top_k(adversary, testdata_tilde, k)
	

# ╔═╡ a8bb0e30-27e4-11eb-1578-53806dd0f0e9
Autoencoder.img(decoder(randn(64)))'

# ╔═╡ 32323f80-27e5-11eb-1581-3537f89b6549
hcat((true_images .- decoded_images)...)

# ╔═╡ 54351b60-27e6-11eb-3898-15504bbff380
Gray.(reshape(collect(testdata_tilde)[6][1], 8, 8))

# ╔═╡ 5c03be90-27e7-11eb-0d07-6529c6633b4f
onehotvector(i, n, v=1; X=zeros(Int, n)) = ((()->X[i]=v)(), X)[end]

# ╔═╡ 4999e350-27e8-11eb-143b-dd46d2a2a8d9
feature_images = Autoencoder.img.([decoder(onehotvector(argmax(softmax(vec(args_x))), 64, 10)) for (args_x, args_y) in collect(testdata_tilde)[args]])'

# ╔═╡ eddba8a0-27e2-11eb-3c2b-456abfb6d91c
likely_failures = hcat(vcat.(true_images, feature_images, decoded_images, failure_indicators)...)

# ╔═╡ dcb73e50-27e6-11eb-3f42-35c9f67c96da
Autoencoder.img(decoder(onehotvector(1,64,10)))'

# ╔═╡ 6ff44460-27e7-11eb-371e-9dddb1d9a76c
onehotvector(1,10)

# ╔═╡ 8adfbf20-27e7-11eb-0cec-91f326f42e55
ex = onehotvector(1,64)

# ╔═╡ 0fa348e0-27e7-11eb-2dc0-a76ea59d1f13
hcat(vcat.([[Autoencoder.img(decoder(onehotvector(i, 64, 10))) for i in (8j - 7):8j] for j in 1:8])...)

# ╔═╡ 0ac1d330-27e9-11eb-148e-156e7d670757
hcat(vcat.([[Autoencoder.img(decoder(onehotvector(i, 64, 10))) for i in (8j - 7):8j] for j in 1:8]...)...)

# ╔═╡ Cell order:
# ╠═4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
# ╠═5c706e30-19b1-11eb-18ad-5190eca820b1
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
# ╠═95bd6740-1a17-11eb-3372-21bac6614f53
# ╠═7b8cf5c0-1a30-11eb-149c-5fbf3a4d9a12
# ╠═a9de0db0-1a2b-11eb-2144-e5025a1a9e5e
# ╠═769264b0-1a30-11eb-3742-7f8badcea17f
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
# ╠═880aede0-27df-11eb-342c-af78db5ae8bb
# ╠═9207b11e-27df-11eb-262b-83039547db91
# ╠═c06496a0-27df-11eb-0557-35927114712c
# ╠═e9f2b600-27df-11eb-3190-d3a359b8b5cb
# ╠═314cdad0-27e0-11eb-261b-359761d4268b
# ╠═1a62c410-27e0-11eb-14ab-339edb9da195
# ╠═9c05b770-27e0-11eb-2116-55c62bb8e032
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
# ╠═fe084142-27e5-11eb-278e-d18844c41964
# ╠═eddba8a0-27e2-11eb-3c2b-456abfb6d91c
# ╠═8c6f5540-27e6-11eb-226b-3d61418cc764
# ╠═a9910dc0-27e2-11eb-1b6c-bf7fbc836b6c
# ╠═a8bb0e30-27e4-11eb-1578-53806dd0f0e9
# ╠═32323f80-27e5-11eb-1581-3537f89b6549
# ╠═54351b60-27e6-11eb-3898-15504bbff380
# ╠═dcb73e50-27e6-11eb-3f42-35c9f67c96da
# ╠═5c03be90-27e7-11eb-0d07-6529c6633b4f
# ╠═6ff44460-27e7-11eb-371e-9dddb1d9a76c
# ╠═8adfbf20-27e7-11eb-0cec-91f326f42e55
# ╠═0fa348e0-27e7-11eb-2dc0-a76ea59d1f13
# ╠═0ac1d330-27e9-11eb-148e-156e7d670757
