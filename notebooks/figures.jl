### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 76b8f2b0-2719-11eb-37f6-01188db45771
using TikzNeuralNetworks

# ╔═╡ 36fce9f0-27c4-11eb-2ee9-bb76b493374e
using PyPlot

# ╔═╡ 643b6720-27c4-11eb-1c36-6f9262dc4995
using Seaborn

# ╔═╡ b5d7170e-2719-11eb-02ca-172fa61417d1
md"""
## Autoencoder
"""

# ╔═╡ 8263b410-2719-11eb-03ad-4fe38be29532
begin
	tikznn = TikzNeuralNetwork(
		input_size=5,
		input_arrows=false,
		input_label=i->i == 3 ? "\$\\mathbf{x}\$" : "",
		# activation_functions=["g","g","g"],0
		hidden_layer_sizes=[4,2,4],
		hidden_layer_labels=["", "\$\\tilde{\\mathbf{x}}\$", ""],
		output_size=5,
		output_arrows=false,
		output_label=i->i == 3 ?  "\$\\mathbf{x}'\$" : "")
	tikznn.tikz.width="12cm"
	tikznn
end

# ╔═╡ 45ff5610-27a9-11eb-1592-69eb568a2e55
save(TEX("../tex/autoencoder-nn.tex"), tikznn)

# ╔═╡ be9ee750-271a-11eb-08aa-9377c292341d
md"""
## Adversary

$$\mathcal{J}(\mathbf{\hat{y}}, \mathbf{y}) = \frac{1}{m}\sum_{i=1}^m \mathcal{L}(\hat{y}_i, y_i)$$


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
		input_size=4,
		input_arrows=false,
		input_label=i->"\$x_{$i}\$",
		hidden_layer_sizes=[2],
		hidden_layer_labels=["ReLU"],
		output_size=1,
		output_arrows=false,
		output_label=i->"\$\\mathbf{\\hat{y}}\$")
	tikzsut.tikz.width="12cm"
	tikzsut
end

# ╔═╡ 3893b690-27c4-11eb-084f-1fe93bb22697
M = [10 0 0 0 0;
     0 10 0 0 0;
     0 0 10 0 0;
     0 0 0 10 0;
     0 0 0 0 10]

# ╔═╡ 99c0e13e-27c4-11eb-0ce5-3521508cfbe4
close("all"); heatmap(M, annot=true, cmap="viridis", xticklabels=1:5, yticklabels=1:5); xlabel(L"x"); ylabel(L"y"); gcf()

# ╔═╡ fbf23580-27c4-11eb-2f92-0bd160f6cdc4
savefig("confusion.pgf")

# ╔═╡ Cell order:
# ╠═76b8f2b0-2719-11eb-37f6-01188db45771
# ╟─b5d7170e-2719-11eb-02ca-172fa61417d1
# ╠═8263b410-2719-11eb-03ad-4fe38be29532
# ╠═45ff5610-27a9-11eb-1592-69eb568a2e55
# ╠═be9ee750-271a-11eb-08aa-9377c292341d
# ╠═c15bbd60-271a-11eb-08f7-2349fae8fd8f
# ╟─45bb62f0-271f-11eb-2462-9bbd423021d4
# ╠═49b91d70-271f-11eb-3c2e-81f1d12885c5
# ╠═36fce9f0-27c4-11eb-2ee9-bb76b493374e
# ╠═643b6720-27c4-11eb-1c36-6f9262dc4995
# ╠═3893b690-27c4-11eb-084f-1fe93bb22697
# ╠═99c0e13e-27c4-11eb-0ce5-3521508cfbe4
# ╠═fbf23580-27c4-11eb-2f92-0bd160f6cdc4
