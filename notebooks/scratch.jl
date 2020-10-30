### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
using FailureRepresentation

# ╔═╡ 5c706e30-19b1-11eb-18ad-5190eca820b1
begin
	using BSON
	using BSON: load
	using CUDA
	using FailureRepresentation.SystemUnderTest
	using FailureRepresentation.Autoencoder
	using Flux
	using Flux: onecold
	using NNlib
	Core.eval(Main, :(import Flux)) # required for .load
end

# ╔═╡ d4fcfdae-1a14-11eb-0a76-5dfe76d1754a
using Plots; plotlyjs()

# ╔═╡ 54164520-19b6-11eb-376c-b5246d05c3a6
MODEL = BSON.load("../models/sut.bson")[:m]

# ╔═╡ 30bd5680-19b2-11eb-3ed2-81593376e639
sut = SystemUnderTest.load("../models/sut.bson")

# ╔═╡ 1732d5e0-19b3-11eb-3b9d-8b5b61fc38ae
_, testdata = SystemUnderTest.getdata();

# ╔═╡ 24ef3930-19b3-11eb-0d9d-e3a0733ba1da
X, Y = rand(testdata, 3)

# ╔═╡ c3996450-1a14-11eb-2b50-934336a06d5a
Xc, Yc = deepcopy(X), deepcopy(Y);

# ╔═╡ b3418cb2-19b3-11eb-25b4-4b48a12ad82e
hcat([Autoencoder.img(X[:,i])' for i in 1:3])

# ╔═╡ 8becbb82-19b3-11eb-2c35-b17e8dee1ed2
onecold(sut(X)) .- 1, onecold(Y) .- 1 # notice off-by-one

# ╔═╡ 708c22b0-19b1-11eb-3ee7-f32bf48ebf1f
autoencoder = Autoencoder.load("../models/autoencoder.bson")

# ╔═╡ 1cef6c80-19b5-11eb-1f69-7356f410b1fc
hcat([Autoencoder.img(autoencoder(X[:,i]))' for i in 1:3])

# ╔═╡ 3cf13db0-19b5-11eb-081b-ed8141b6b8d7
onecold(sut(autoencoder(X))) .- 1, onecold(Y) .- 1 # notice off-by-one

# ╔═╡ f4fbfc40-19b2-11eb-0ee1-0b3e9ece6008
Autoencoder.sample(autoencoder, 3)

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

# ╔═╡ 805fa7c0-1a15-11eb-2200-a166cafe9431
# function Δsoftmax_expanded(model, x, y)
# 	probs = softmax(model(x))
# 	ŷᵢₙₓ = onecold(probs)
# 	yᵢₙₓ = onecold(y)
# 	max_softmax = [probs[r,c] for (r,c) in zip(onecold(probs), 1:size(probs,2))]
# 	true_softmax = [probs[r,c] for (r,c) in zip(onecold(Yi), 1:size(probs,2))]
# 	return max_softmax - true_softmax
# end;

# ╔═╡ 99924160-1a35-11eb-0df0-ff718845bd4b
Xi, Yi = rand(testdata, 10)

# ╔═╡ 7b8cf5c0-1a30-11eb-149c-5fbf3a4d9a12
Δsoftmax(sut, Xi[:, 6], Yi[:, 6])

# ╔═╡ 769264b0-1a30-11eb-3742-7f8badcea17f
Δsoftmax(sut, Xi, Yi)

# ╔═╡ 009237b0-1a15-11eb-3e6c-9ff29ea9a32a
begin
	idx = 6
	bar(0:9, Yi[:,idx], xticks=0:9, fillalpha=0.6, label="true")
	bar!(0:9, softmax(sut(Xi[:,idx])), fillalpha=0.6, label="prediction")
end

# ╔═╡ e083cc00-1ac7-11eb-103a-13fb6fb3d586
Autoencoder.img(Xi[:,idx])'

# ╔═╡ 9f53f2fe-1a35-11eb-241a-15d6dc131650
Δsoftmax(sut, Xi, Yi)

# ╔═╡ Cell order:
# ╠═4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
# ╠═5c706e30-19b1-11eb-18ad-5190eca820b1
# ╠═54164520-19b6-11eb-376c-b5246d05c3a6
# ╠═30bd5680-19b2-11eb-3ed2-81593376e639
# ╠═1732d5e0-19b3-11eb-3b9d-8b5b61fc38ae
# ╠═24ef3930-19b3-11eb-0d9d-e3a0733ba1da
# ╠═c3996450-1a14-11eb-2b50-934336a06d5a
# ╠═b3418cb2-19b3-11eb-25b4-4b48a12ad82e
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
# ╠═805fa7c0-1a15-11eb-2200-a166cafe9431
# ╠═009237b0-1a15-11eb-3e6c-9ff29ea9a32a
# ╠═e083cc00-1ac7-11eb-103a-13fb6fb3d586
# ╠═99924160-1a35-11eb-0df0-ff718845bd4b
# ╠═9f53f2fe-1a35-11eb-241a-15d6dc131650
