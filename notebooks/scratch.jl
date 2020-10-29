### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
using FailureRepresentation

# ╔═╡ 5c706e30-19b1-11eb-18ad-5190eca820b1
begin
	using BSON: load
	using CUDA
	using FailureRepresentation.SystemUnderTest
	using FailureRepresentation.Autoencoder
	using Flux
	using Flux: onecold
	using NNlib
	Core.eval(Main, :(import Flux)) # required for .load
end

# ╔═╡ 30bd5680-19b2-11eb-3ed2-81593376e639
sut = SystemUnderTest.load("../models/sut.bson")

# ╔═╡ 1732d5e0-19b3-11eb-3b9d-8b5b61fc38ae
_, testdata = SystemUnderTest.getdata();

# ╔═╡ 24ef3930-19b3-11eb-0d9d-e3a0733ba1da
X, Y = rand(testdata, 3)

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

# ╔═╡ Cell order:
# ╠═4af6dbd0-19b1-11eb-059b-c7ac5d8c4e0d
# ╠═5c706e30-19b1-11eb-18ad-5190eca820b1
# ╠═30bd5680-19b2-11eb-3ed2-81593376e639
# ╠═1732d5e0-19b3-11eb-3b9d-8b5b61fc38ae
# ╠═24ef3930-19b3-11eb-0d9d-e3a0733ba1da
# ╠═b3418cb2-19b3-11eb-25b4-4b48a12ad82e
# ╠═8becbb82-19b3-11eb-2c35-b17e8dee1ed2
# ╠═708c22b0-19b1-11eb-3ee7-f32bf48ebf1f
# ╠═1cef6c80-19b5-11eb-1f69-7356f410b1fc
# ╠═3cf13db0-19b5-11eb-081b-ed8141b6b8d7
# ╠═f4fbfc40-19b2-11eb-0ee1-0b3e9ece6008
