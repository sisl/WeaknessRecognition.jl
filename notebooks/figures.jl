### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 76b8f2b0-2719-11eb-37f6-01188db45771
using TikzNeuralNetworks

# ╔═╡ b5d7170e-2719-11eb-02ca-172fa61417d1
md"""
## Autoencoder
"""

# ╔═╡ 8263b410-2719-11eb-03ad-4fe38be29532
begin
	tikznn = TikzNeuralNetwork(
		input_size=5,
		input_arrows=false,
		input_label=i->"\$x_{$i}\$",
		hidden_layer_sizes=[4,2,4],
		hidden_layer_labels=["", "Low-dim.", ""],
		output_size=5,
		output_arrows=false,
		output_label=i->"\$\\hat{y}_{$i}\$")
	tikznn.tikz.width="12cm"
	tikznn
end

# ╔═╡ be9ee750-271a-11eb-08aa-9377c292341d
md"""
## Adversary

\$\\mathcal{J}(\\mathbf{\\hat{y}}, \\mathbf{y}) = \\frac{1}{m}\\sum \\mathcal{L}(\\hat{y}, y)\$


$$\mathcal{L}(\hat{y}, y) = - y \log(\hat y) - (1 - y)\log(1-\hat y)$$

$$\mathcal{L}(\hat{y}, y) = - y \log(\hat y) \Omega - (1 - y)\log(1-\hat y)$$


"""

# ╔═╡ c15bbd60-271a-11eb-08f7-2349fae8fd8f
begin
	tikzadv = TikzNeuralNetwork(
		input_size=2,
		input_arrows=false,
		input_label=i->"\$x_{$i}\$",
		hidden_layer_sizes=[4,2],
		hidden_layer_labels=["ReLU", "ReLU"],
		output_size=1,
		output_arrows=false,
		output_label=i->"\$\\hat{y}_{$i}\$")
	tikzadv.tikz.width="12cm"
	tikzadv
end

# ╔═╡ 45bb62f0-271f-11eb-2462-9bbd423021d4
md"""
## System Under Test
"""

# ╔═╡ 49b91d70-271f-11eb-3c2e-81f1d12885c5
begin
	tikzsut = TikzNeuralNetwork(
		input_size=5,
		input_arrows=false,
		input_label=i->"\$x_{$i}\$",
		hidden_layer_sizes=[3],
		hidden_layer_labels=["ReLU"],
		output_size=5,
		output_arrows=false,
		output_label=i->"\$\\hat{y}_{$i}\$")
	tikzsut.tikz.width="12cm"
	tikzsut
end

# ╔═╡ Cell order:
# ╠═76b8f2b0-2719-11eb-37f6-01188db45771
# ╟─b5d7170e-2719-11eb-02ca-172fa61417d1
# ╠═8263b410-2719-11eb-03ad-4fe38be29532
# ╠═be9ee750-271a-11eb-08aa-9377c292341d
# ╠═c15bbd60-271a-11eb-08f7-2349fae8fd8f
# ╠═45bb62f0-271f-11eb-2462-9bbd423021d4
# ╠═49b91d70-271f-11eb-3c2e-81f1d12885c5
