### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 4cc97f4d-7a4c-487a-8684-1edd1bb963a5
begin
	using LinearAlgebra, Random, LaTeXStrings
	using Plots, Plots.PlotMeasures, PlutoUI, Images
	using MLDatasets, Enzyme ,  Statistics
end 

# ╔═╡ ddf6ac4d-df08-4e73-bc60-4925aa4b94c8
md"""
# Implementing a Vision Transformer (ViT) in Julia!

The Transformer architecture, introduced in the paper _Attention Is All You Need_ by [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), is the most ubiquitous neural network architecture in modern machine learning. Its parallelism and scalability to large problems has seen it adopted in domains beyong those it was traditionally considered for (sequential data). 

![ViT Model](https://github.com/qsimeon/julia_class_project/blob/main/transformer_architecture.jpg?raw=true)

**NOTE:** We adapt/borrow a lot of material/concepts from
[Torralba, A., Isola, P., & Freeman, W. T. (2021, December 1). _Foundations of Computer Vision_. MIT Press; The MIT Press, Massachusetts Institute of Technology.](https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/)

---
"""

# ╔═╡ 970f0e2b-459b-4baa-ae30-886c2bada7b4

md"""
One thing to keep in mind throughout this notebook is that Transformers operate on **tokens**. Conceptually, the transformer architecture may be thought of as a _token net_, which is abtraction or generalization of the more familiar _neural nets_.

Token nets are just like neural nets, alternating between layers that mix nodes in linear combinations (e.g., fully connected linear layers, convolutional layers, etc.) and layers that apply a pointwise nonlinearity to each node (e.g., relus, per-token MLPs). 

![Token Nets](https://github.com/qsimeon/julia_class_project/blob/main/token_net.jpg?raw=True)

---
"""

# ╔═╡ 191c435c-4094-4326-9e18-0ee8dc3058ab
md"""
Until recently, the best performing models for image classification had been convolutional neural networks (CNNs) introduced in [LeCun et al. (1998)](https://ieeexplore.ieee.org/abstract/document/726791). Nowadays, transformer architectures have been shown to have similar to better performance. One such model, called Vision Transformer by [Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929) splits up images into regularly sized patches. The patches are treated as a sequence and attention weights are learned as in a standard transformer model.

![ViT Model](https://github.com/qsimeon/julia_class_project/blob/main/vit_architecture.jpg?raw=true)

---
"""

# ╔═╡ 2348f0c3-5fc1-424f-8a56-c00c52ca9a4f
md"""
Let’s start by defining key components of a **Transformer** model using Julia structs and parametric types, similar to the structure we implemented in *Homework 3*. 

We will implement the `AttentionHead`, `MultiHeadedAttention`, and `FeedForwardNetwork` modules as Julia structs. This will set up the parts which get combined together in the `Transformer` model.

---
"""

# ╔═╡ afe50e6c-9e61-4246-a8ac-bebc83e2715c
# Stable softmax implementation
function softmax(x; dims=1)
    exp_x = exp.(x .- maximum(x, dims=dims))  # stability trick
    return exp_x ./ sum(exp_x, dims=dims)
end

# ╔═╡ ddc663b2-9de3-11ef-1d3a-9f172c4dda5f
### 1. Attention Head
struct AttentionHead{T<:Real}
    W_K::Matrix{T} # Shape: (n_hidden, dim)
    W_Q::Matrix{T} # Shape: (n_hidden, dim)
    W_V::Matrix{T} # Shape: (dim, dim)
    n_hidden::Int # dimensionality of key and query vectors

    function AttentionHead{T}(dim::Int, n_hidden::Int) where T<:Real
        return new{T}(randn(T, n_hidden, dim), randn(T, n_hidden, dim), randn(T, dim, dim), n_hidden)
    end

    function (head::AttentionHead{T})(X::Array{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
		# X is expected to be an input token matrix with shape (N, dim)
        # Project input tokens to query, key, and value representations
        Q = X * transpose(head.W_Q)  # Shape: (N, n_hidden)
        K = X * transpose(head.W_K)  # Shape: (N, n_hidden)
        V = X * transpose(head.W_V)  # Shape: (N, dim)
        
        # Compute scaled dot-product attention
        scores = Q * transpose(K) / sqrt(head.n_hidden)  # Shape: (N, N)
        
        # Apply attention mask if provided
        if attn_mask !== nothing
            scores = scores .* attn_mask .+ (1 .- attn_mask) * -Inf
        end
        
        # Apply softmax along the last dimension
        alpha = softmax(scores, dims=ndims(scores))  # Shape: (N, N)
        
        # Compute attention output as weighted sum of values
        attn_output = alpha * V  # Shape: (N, dim)

		# attn_output is the (N, dim) output token matrix
		# alpha is the (N, N) attention matrix
        return attn_output, alpha
    end
end


# ╔═╡ 9adfff6a-e83e-4266-8bae-67f4a16e011f
@bind n_tokens Slider(5:20, show_value=true)

# ╔═╡ 5a498179-0be9-4e70-988f-14575d12a396
# Test `AttentionHead` implementation
let
	dim, attn_dim = 3, 8
	head = AttentionHead{Float64}(dim, attn_dim)
	X = randn(Float64, n_tokens, dim)  # example 3-D input of n_tokens
	attn_output, alpha = head(X)
	println("attention output shape: ", size(attn_output))
	println("attention weight shape: ", size(alpha))
	heatmap(
	    alpha,
	    aspect_ratio=:equal,
		xlabel="Token Index",
    	ylabel="Token Index",
    	title="Attention Matrix",
	    c=:plasma,                      # Choose a colormap, e.g., :viridis or :plasma
		
	    clabel="Weight",                 # Label for the color bar
	    colorbar=true,                   # Show the color bar
	    grid=false,                      # Turn off the grid
		# framestyle=:none,         # Removes the axis lines
	    xticks=1:n_tokens,  # Ensures ticks are at each integer index
	    yticks=1:n_tokens,
	)

end

# ╔═╡ c3eaadcf-a06d-4469-ba9a-399043e72a9f
md"""
To compute the query, key, and value for a set of input tokens, $\mathbf{T}_{\text {in }}$, we apply the same linear transformations to each token in the set, resulting in matrices $\mathbf{Q}_{\mathrm{in}}, \mathbf{K}_{\mathrm{in}} \in \mathbb{R}^{N \times m}$ and $\mathbf{V}_{\text {in }} \in \mathbb{R}^{N \times d}$, where each row is the query/key/value for each token:

```math
\begin{aligned}
& \mathbf{Q}_{\text {in }}=\left[\begin{array}{c}
\mathbf{q}_1^{\top} \\
\vdots \\
\mathbf{q}_N^{\top}
\end{array}\right]=\left[\begin{array}{c}
\left(\mathbf{W}_q \mathbf{t}_1\right)^{\top} \\
\vdots \\
\left(\mathbf{W}_q \mathbf{t}_N\right)^{\top}
\end{array}\right]=\mathbf{T}_{\text {in }} \mathbf{W}_q^{\top} \quad \triangleleft \quad \text {query matrix} 
\\
& \mathbf{K}_{\text {in }}=\left[\begin{array}{c}
\mathbf{k}_1^{\top} \\
\vdots \\
\mathbf{k}_N^{\top}
\end{array}\right]=\left[\begin{array}{c}
\left(\mathbf{W}_k \mathbf{t}_1\right)^{\top} \\
\vdots \\
\left(\mathbf{W}_{\mathrm{k}} \mathbf{t}_N\right)^{\top}
\end{array}\right]=\mathbf{T}_{\text {in }} \mathbf{W}_k^{\top} \quad \triangleleft \quad \text { key matrix }
\\
& \mathbf{V}_{\text {in }}=\left[\begin{array}{c}
\mathbf{v}_1^{\top} \\
\vdots \\
\mathbf{v}_N^{\top}
\end{array}\right]=\left[\begin{array}{c}
\left(\mathbf{W}_v \mathbf{t}_1\right)^{\top} \\
\vdots \\
\left(\mathbf{W}_v \mathbf{t}_N\right)^{\top}
\end{array}\right]=\mathbf{T}_{\text {in }} \mathbf{W}_v^{\top} \quad \triangleleft \quad \text { value matrix }
\end{aligned}
```

> Note that the query and key vectors must have the same dimensionality, $m$, because we take a dot product between them. Conversely, the value vectors must match the dimensionality of the token code vectors, $d$, because these are summed up to produce the new token code vectors.

Finally, we have the attention equation:

```math
\begin{aligned}
\mathbf{A} & = \text { softmax }\left(\frac{\mathbf{Q}_{\mathrm{in}} \mathbf{K}_{\mathrm{in}}^{\top}}{\sqrt{m}}\right) \quad \triangleleft \quad \text { attention matrix } \\
\mathbf{T}_{\text {out }} & =\mathbf{A} \mathbf{V}_{\text {in }}
\end{aligned}
```

where the softmax is taken within each row (i.e., over the vector of matches for each separate query vector.
"""

# ╔═╡ 245ce308-8fc2-4b31-8aa6-d7c1d33b61ca
# Showing causal attention with mask
let
	dim, attn_dim = 3, 8
	args = (aspect_ratio=1, colorbar=false, grid=false, yticks=false, xticks=false, size=(650,650))
	
	X = randn(Float64, n_tokens, dim)  # example 3-D input of n_tokens
	W_Q = randn(n_tokens, dim)
	W_K = randn(n_tokens, dim)
	W_V = randn(dim, dim)

	p1 = heatmap(X, title=string("in tokens ", L"T_{in}"); args..., ylabel=string("position, ", L"L"), xlabel=string("feature, ", L"d"))
	p2 = heatmap(W_Q, title=L"W_q"; args...)
	p3 = heatmap(W_K, title=L"W_k"; args...)
	p4 = heatmap(W_V, title=L"W_v"; args...)

	Q = X * transpose(W_Q)
	K = X * transpose(W_K)
	V = X * transpose(W_V)
	
	p5 = heatmap(Q, title=L"Q = T_{in} W_q^\intercal"; args...)
	p6 = heatmap(K, title=L"K = T_{in} W_k^\intercal"; args...)
	p7 = heatmap(V, title=L"V = T_{in} W_v^\intercal"; args...)
	
	scores = Q * transpose(K) / sqrt(attn_dim) # n_tokens x n_tokens
	@assert size(scores) == (n_tokens, n_tokens)

	p8 = heatmap(scores, title="scores"; args...)

	mask = UpperTriangular(ones(n_tokens, n_tokens))
	
	p9 = heatmap(mask, title="causal mask"; args...)
	
	masked_scores = scores .* mask .+ (1 .- mask) 
	
	p10 = heatmap(masked_scores, title="mask scores"; args...)

	alpha = softmax(masked_scores, dims=ndims(masked_scores)) 
	
	p11 = heatmap(alpha, title=string("attention ", L"A"); args...)

	attn_output = alpha * V
	
	p12 = heatmap(attn_output, title=string("out tokens ", L"T_{out}"); args...)

	plot!([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]..., layout=(3,4))
end

# ╔═╡ 1c2692a1-e8a0-4926-ad42-3787671eeb51
md"""
In expanded detail, here are the full mechanics of a self-attention layer, which is the kind of attention layer used in transformers. 

![Self-Attention Layer](https://github.com/qsimeon/julia_class_project/blob/main/self_attention_layer.jpg?raw=true)
"""

# ╔═╡ 74eb85f0-ea48-48cd-b732-4d97f4883c85
### 2. Multi-Headed Attention
struct MultiHeadedAttention{T<:Real}
    heads::Vector{AttentionHead{T}}
    W_msa::Matrix{T} # Shape: (dim, num_heads*dim)

    function MultiHeadedAttention{T}(dim::Int, n_hidden::Int, num_heads::Int) where T<:Real
		# Each head outputs dim-dimensional tokens
        heads = [AttentionHead{T}(dim, n_hidden) for _ in 1:num_heads]
		# Our MHA outputs tokens with the same dimension as the input tokens
        W_msa = randn(T, dim, num_heads * dim)
        return new{T}(heads, W_msa)
    end

    function (mha::MultiHeadedAttention{T})(X::Array{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
        outputs, alphas = [], []
        for head in mha.heads
            out, alpha = head(X, attn_mask) # Shapes: (N, dim), (N, N) 
            push!(outputs, out)
            push!(alphas, alpha)
        end
		# Concatenate along hidden dimension
        concatenated = cat(outputs...; dims=2) # Shape: (N, num_heads*dim)
        attn_output = concatenated * transpose(mha.W_msa) # Shape: (N, dim)
		attn_alphas = cat(alphas...; dims=3) # Shape: (N, N, num_heads)
		attn_alphas = permutedims(attn_alphas, (3, 1, 2)) # Shape: (num_heads, N, N)
        return attn_output, attn_alphas
    end
end

# ╔═╡ c8d32f75-83a3-40d7-b136-4bf5966612a0
# Test `MultiHeadedAttention` implementation
let
	dim, attn_dim, num_heads = 3, 8, 5
	heads = [AttentionHead{Float64}(dim, attn_dim) for _ in 1:num_heads]
	W_msa = randn(Float64, dim, num_heads * dim)
	X = randn(Float64, n_tokens, dim)  # example 3-D input of n_tokens
	
	outputs, alphas = [], []
	for head in heads
		attn_out, alpha = head(X)
		push!(outputs, attn_out)
		push!(alphas, alpha)
	end

	concatenated = cat(outputs...; dims=2) # Shape: (N, num_heads*dim)
	attn_alphas = cat(alphas...; dims=3) # Shape: (N, N, num_heads)
	attn_alphas = permutedims(attn_alphas, (3, 1, 2)) # Shape: (num_heads, N, N)
	attn_output = concatenated * transpose(W_msa) # Shape: (N, dim)

	println("attention output shape: ", size(attn_output))
	println("attention weights shape: ", size(attn_alphas))
	# plot the attention mask from the last head
	heatmap(
	    attn_alphas[end,:,:],
	    aspect_ratio=:equal,
		xlabel="Token Index",
    	ylabel="Token Index",
    	title="Attention Matrix (last head)",
	    c=:plasma,                      # Choose a colormap, e.g., :viridis or :plasma
	    clabel="Weight",                 # Label for the color bar
	    colorbar=true,                   # Show the color bar
	    grid=false,                      # Turn off the grid
		# framestyle=:none,         # Removes the axis lines
	    xticks=1:n_tokens,  # Ensures ticks are at each integer index
	    yticks=1:n_tokens,
	)
end

# ╔═╡ e9ce397a-b315-470d-9534-cdd1dad12a1b
# key / query demonstration
let
    url = "https://github.com/qsimeon/julia_class_project/blob/main/attention_map.png?raw=true"
    imf = download(url)
    im = load(imf)
end

# ╔═╡ 661d546e-4182-4199-9b2d-3c5eb90bc07f
### 3. Feed-Forward Network (FFN)
struct Linear{T<:Real}
    W::Matrix{T}  # Weight matrix
    b::Vector{T}  # Bias vector

    # Constructor: Initialize weights and biases
    function Linear{T}(in_features::Int, out_features::Int) where T<:Real
        W = randn(T, out_features, in_features) / sqrt(in_features)  # Xavier initialization
        b = zeros(T, out_features)
        return new{T}(W, b)
    end

    # Apply the linear transformation
    function (linear::Linear{T})(X::Array{T}) where T<:Real
        @assert size(X, 2) == size(linear.W, 2) "Input dimension must match the weight matrix"
        return X * transpose(linear.W) .+ linear.b'
    end
end

# ╔═╡ 03759beb-1ac3-4bd7-800b-f4461edb58b1
# Test `Linear` implementation
let
    in_features = 64   # Input dimension
    out_features = 10  # Output dimension

    # Initialize the Linear module
    linear_layer = Linear{Float64}(in_features, out_features)

    # Example input matrix (T, in_features), where T is the number of tokens
    n_tokens = 20  # Number of tokens
    X = randn(Float64, n_tokens, in_features)

    # Apply the linear transformation
    output = linear_layer(X)

    # Verify dimensions
    println("Input shape: ", size(X))        # Should be (n_tokens, in_features)
    println("Output shape: ", size(output))  # Should be (n_tokens, out_features)
end

# ╔═╡ c12ffac7-7533-42fa-a721-30938396e898
### 4. Feed-Forward Network (FFN)
struct FeedForwardNetwork{T<:Real}
    layer1::Linear{T}  # First linear transformation
    layer2::Linear{T}  # Second linear transformation

    function FeedForwardNetwork{T}(dim::Int, n_hidden::Int) where T<:Real
        # Initialize the two linear layers
        layer1 = Linear{T}(dim, n_hidden)  # Shape: (n_hidden, dim)
        layer2 = Linear{T}(n_hidden, dim)  # Shape: (dim, n_hidden)
        return new{T}(layer1, layer2)
    end

    function (ffn::FeedForwardNetwork{T})(X::Array{T}) where {T<:Real}
        # Apply the first linear transformation
        X = ffn.layer1(X)  # Shape: (N, n_hidden)
        X = max.(0, X)     # ReLU activation
        # Apply the second linear transformation
        return ffn.layer2(X)  # Shape: (N, dim)
    end
end

# ╔═╡ a7f306c1-12aa-4ecc-9764-70a72c41bd67
# Test `FeedForwardNetwork` implementation
let
    dim, mlp_dim = 3, 8
    ffn = FeedForwardNetwork{Float64}(dim, mlp_dim)
    X = randn(Float64, n_tokens, dim)  # Example input with n_tokens tokens of dim-dimensional vectors
    ffn_output = ffn(X)
    
    # Verify output shape
    println("Input shape: ", size(X))          # Should be (n_tokens, dim)
    println("Output shape: ", size(ffn_output)) # Should be (n_tokens, dim)
end


# ╔═╡ 2f40b3cf-e690-40be-8e5c-b66e022c505d
md"""
## Recap so far

1. **`AttentionHead` Implementation**:
    - Projects the input token matrix `X` (shape: $$(N, \text{dim})$$) to query, key, and value matrices.
    - Computes scaled dot-product attention, applies an optional mask, and then applies softmax to get attention weights `alpha` with shape $$(N, N)$$.
    - Returns the attention output `attn_output` (shape: $$(N, \text{dim})$$) and attention weights `alpha`.

2. **`MultiHeadedAttention` Implementation**:
    - Creates multiple `AttentionHead` instances and collects their outputs.
    - Concatenates these outputs along the hidden dimension, applies a linear transformation (`W_msa`), and stacks the attention weights from each head into a 3D tensor with shape `(num_heads, N, N)`.

3. **`FeedForwardNetwork` (FFN) Implementation**:
    - A two-layer feed-forward network with an intermediate hidden layer of size `n_hidden`.
    - Projects the input token matrix `X` (shape: $$(N, \text{dim})$$) to an intermediate hidden representation (shape: $$(N, \text{n\_hidden})$$) using `W1` and `b1`, followed by a ReLU activation.
    - Transforms the hidden representation back to the original input dimension `dim` using `W2` and `b2`.
    - Returns the output with shape $$(N, \text{dim})$$, maintaining the same dimension as the input tokens.
"""


# ╔═╡ 97c1b967-b634-4ff0-8007-939bf8ea87fa
### 5. Attention Residual
struct AttentionResidual{T<:Real}
	attn::MultiHeadedAttention{T}  # Multi-headed attention mechanism
	ffn::FeedForwardNetwork{T}      # Feed-forward network

	# Constructor: initializes attention and feed-forward sub-layers
	function AttentionResidual{T}(dim::Int, attn_dim::Int, mlp_dim::Int, num_heads::Int) where T<:Real
		attn_layer = MultiHeadedAttention{T}(dim, attn_dim, num_heads)
		ffn_layer = FeedForwardNetwork{T}(dim, mlp_dim)
		return new{T}(attn_layer, ffn_layer)
	end

	# Apply the AttentionResidual block to input x
	function (residual::AttentionResidual{T})(X::Array{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
		# Apply the multi-headed attention layer
		attn_out, alphas = residual.attn(X, attn_mask)  # attn_out: (N, dim), alphas: (num_heads, N, N)
		# First residual connection with attention output
		X = X .+ attn_out
		# Apply the feed-forward network and add the second residual connection
		X = X .+ residual.ffn(X)
		# Return the final output and attention weights
		return X, alphas
	end
end


# ╔═╡ 9bd646c3-ef9d-4b06-a598-267c0cbdff4a
### 6. Transformer
struct Transformer{T<:Real}
    layers::Vector{AttentionResidual{T}}  # Sequence of AttentionResidual blocks

    # Constructor: initializes a sequence of attention residual blocks
    function Transformer{T}(dim::Int, attn_dim::Int, mlp_dim::Int, num_heads::Int, num_layers::Int) where T<:Real
        layers = [AttentionResidual{T}(dim, attn_dim, mlp_dim, num_heads) for _ in 1:num_layers]
        return new{T}(layers)
    end

    # Apply the Transformer model to input X
    function (transformer::Transformer{T})(X::Array{T}; attn_mask::Union{Nothing, Matrix{T}}=nothing, return_attn::Bool=false) where T<:Real
        collected_alphas = []  # To store attention weights from each layer
        for layer in transformer.layers
            X, alphas = layer(X, attn_mask)  # Apply each residual block
            if return_attn
                push!(collected_alphas, alphas)  # Collect attention weights
            end
        end
        # Return the final output and collected attention weights from all layers (if required)
        if return_attn
            return X, collected_alphas
        else
            return X, nothing
        end
    end
end


# ╔═╡ e88482de-8685-47d7-9cbb-78328eed8244
md"""
## Testing the `AttentionResidual` and `Transformer`

Let's test the `AttentionResidual` and `Transformer` structs to confirm that they work as expected with the previously implemented components.
"""

# ╔═╡ da2f6c55-17c5-495c-8ba3-5b2dc50a17f1
# Test `AttentionResidual` implementation
let
	dim, attn_dim, mlp_dim, num_heads = 8, 16, 32, 3
	residual_block = AttentionResidual{Float64}(dim, attn_dim, mlp_dim, num_heads)
	X = randn(Float64, n_tokens, dim)  # example input with n_tokens, each of `dim` dimensions
	output, alphas = residual_block(X)
	println("AttentionResidual output shape: ", size(output))
	println("Attention weights shape (from one layer): ", size(alphas))
end

# ╔═╡ fd2a796f-b683-4792-a976-b4071fda58a0
# Test `Transformer` implementation
let
	dim, attn_dim, mlp_dim, num_heads, num_layers = 8, 16, 32, 3, 6
	transformer = Transformer{Float64}(dim, attn_dim, mlp_dim, num_heads, num_layers)
	X = randn(Float64, n_tokens, dim)  # example input with n_tokens, each of `dim` dimensions
	output, collected_alphas = transformer(X; return_attn=true)
	println("Transformer output shape: ", size(output))
	println("Collected attention weights shape ($num_layers layers): ", [size(collected_alphas[i]) for i in 1:length(collected_alphas)])
end

# ╔═╡ 7eb8e4ec-80ae-4744-b21d-8b36885ff98c
md"""
## Our modules so far build up the Transfomer

- **AttentionHead**: Implements a single attention head, creating query, key, and value projections, computing the attention scores, and applying a softmax.
- **MultiHeadedAttention**: Combines multiple `AttentionHead`s, concatenates their outputs, and applies a final linear transformation.
- **FeedForwardNetwork**: A simple feed-forward network with two linear layers and a ReLU activation in between.
- **AttentionResidual**: Combines multi-head attention and feed-forward network layers with residual connections.
- **Transformer**: Stacks multiple `AttentionResidual` layers to form the complete Transformer encoder.

---
"""

# ╔═╡ 90019339-1fdf-4541-b71b-a00b9ef7d904
md"""
Using callable structs, parametric types, and matrix operations, we set up the basic components and combined them to create a Transformer module . 

Let's view our Julia callable struct implemenations of the `AttentionHead` and `Transformer` modules side-by-side with a bare-bones implementation in PyTorch.

__Show side-by-side comparison:__

![AttentionHead PyJl](https://github.com/qsimeon/julia_class_project/blob/main/attention_python_julia.jpg?raw=true)

![Transformer PyJl](https://github.com/qsimeon/julia_class_project/blob/main/transformer_python_julia.jpg?raw=true)

---
"""

# ╔═╡ db19fde5-434c-4030-9a61-0200ddca659f
md"""
But recall that what we want is to make is a **Vision Transformer**. This requires some additional layers for tokenizing an image. These are
* (1) patch embedding; and
* (2) positional encoding

![Patch Embed Image](https://github.com/qsimeon/julia_class_project/blob/main/patch_position.jpg?raw=true)

We will explore these in detail next.

---
"""

# ╔═╡ a2bf29b4-174d-42e1-94b6-39822556349c
md"""
## Patch Embedding

It turns out the patch embedding can be implemented by applying a strided convolution. However, we will take the more direct and visualizable approach of chopping up an image into patches and linearly projecting the vector that is the flattened patch to the desired dimensionality. 

Remember Transformers operate on tokens i.e. transformations of tokens. What we are doing here is essentially the first step of *tokenizing* our image data.

![Patch Embed](https://github.com/qsimeon/julia_class_project/blob/main/patch_tokenize.jpg?raw=true)

---
"""

# ╔═╡ ffeafe79-65b5-4c75-aaf1-e83bc8ca17cc
# Define function to extract patches
function extract_patches(image_square, patch_size)
    patches = []
    for i in 1:patch_size:size(image_square, 1)
        for j in 1:patch_size:size(image_square, 2)
            patch = view(image_square, i:i+patch_size-1, j:j+patch_size-1)
            push!(patches, patch)
        end
    end
    return patches
end

# ╔═╡ c3d2a61c-2caa-4f52-acab-8a0b89e5aac5
function visualize_patches(image_square, patch_size)
    # Extract patches from image
    patches = extract_patches(image_square, patch_size)
    
    # Calculate grid dimensions (assuming a square grid)
    n_patches = length(patches)
    grid_dim = ceil(Int, sqrt(n_patches))
    
    # Initialize a list to hold the individual patch plots
    plot_list = []
    
    # Generate the grid of patches
    for idx in 1:n_patches
        # Create a heatmap for each patch without axis or colorbar
        p = heatmap(patches[idx], color=:grays, axis=false, colorbar=false)
        push!(plot_list, p)
    end
    
    # Display the grid of patches with custom grid padding
    plot(
        plot_list...,
        layout=(grid_dim, grid_dim),
		margin=0mm,
        size=(900, 900),
    )
end

# ╔═╡ 9e705315-d646-4373-854d-47a9f9d9076b
# Load and preprocess an example image
begin
	# Download image of Philip
	url = "https://user-images.githubusercontent.com/6933510/107239146-dcc3fd00-6a28-11eb-8c7b-41aaf6618935.png"  # high-res
    philip_filename = download(url)
	philip = load(philip_filename)

    # Resize the image to a square (e.g., 256x256)
    img_size = 256
    image_square = imresize(philip, (img_size, img_size))
end

# ╔═╡ 5a0607bc-bf03-4b19-894f-1bcfd68a0762
begin
	# Calculate divisors of the image size
	function divisors_of_half_image_size(img_size)
	    divisors = []
	    for i in div(img_size, 16):div(img_size, 1)
	        if img_size % i == 0
	            push!(divisors, i)
	        end
	    end
	    return divisors
	end
	
	# Get the divisors of the image size
	patch_size_options = divisors_of_half_image_size(img_size)
end


# ╔═╡ b35b28dd-e992-49a0-8e01-abd3e26ad093
# Create the slider with the patch size options
@bind patch_size Slider(patch_size_options, show_value=true, default=patch_size_options[1])

# ╔═╡ 8a0abb17-b7f0-4952-b5c1-0d52095cf2bf
# Visualize patches for the chosen `patch_size`
visualize_patches(image_square, patch_size)

# ╔═╡ 44f39ba0-68e6-450d-a7fa-99f180a48b67
md"""
A patch embedding layer is one that takes each of the image patches, like those displayed above, and then projects that into a vector. One approach is to simply flatten each patch and use a linear projection (using a matrix multiplication) to convert this into a vector. Since we are working on RGB images (3 channels), we define a linear projection for each channel independently and then combine them.

---
"""

# ╔═╡ a2ff04a3-4118-47b8-b768-fc2a4986167b
### 7. PatchEmbedLinear 
struct PatchEmbedLinear{T<:Real}
    img_size::Int
    patch_size::Int
    nin::Int  # Number of input channels (e.g., RGB → nin = 3)
    nout::Int # Desired output dimensionality for each patch
    num_patches::Int
    W::Vector{Matrix{T}} # Linear projection weights, one for each channel

    function PatchEmbedLinear{T}(img_size::Int, patch_size::Int, nin::Int, nout::Int) where T<:Real
        @assert img_size % patch_size == 0 "img_size must be divisible by patch_size"
        num_patches = (img_size ÷ patch_size)^2
        # Create a distinct weight matrix for each channel
        W = [randn(T, nout, patch_size^2) for _ in 1:nin]
        return new{T}(img_size, patch_size, nin, nout, num_patches, W)
    end

    function (embed::PatchEmbedLinear{T})(image::Matrix{<:RGB}) where T<:Real
        # Ensure image size matches expected dimensions
        img_size = size(image, 1)
        @assert img_size == embed.img_size "Image size does not match module configuration"

        # Split the RGB image into three separate channel matrices
        channels = channelview(image)  # Shape: (3, 256, 256)
        @assert size(channels, 1) == embed.nin "Number of image channels does not match nin"

        # Extract patches and project for each channel
        projected_channels = []
        for c in 1:embed.nin
            # Extract the channel matrix (2D slice)
            channel_matrix = Matrix{T}(channels[c, :, :])  # Shape: (256, 256)
            # Extract patches
            patches = extract_patches(channel_matrix, embed.patch_size)
            # Flatten patches and project
            patch_matrix = hcat([vec(patch) for patch in patches]...)'  # Shape: (num_patches, patch_size^2)
            projected = patch_matrix * transpose(embed.W[c])  # Shape: (num_patches, nout)
            push!(projected_channels, projected)
        end

        # Sum the projected embeddings across channels
        projected_patches = reduce(+, projected_channels)  # Shape: (num_patches, nout)

        return projected_patches
    end
end


# ╔═╡ 9d6cd065-5f25-4943-b155-3602db474bff
@bind nout Slider([16:16:96...], show_value=true, default=32)

# ╔═╡ 02fb8ff3-647e-4d55-8c2b-a1d9066338ed
# Test `PatchEmbedLinear` implementation for RGB images
let
    img_size = size(image_square, 1) # Image size (assumes square image: img_size x img_size)
    nin = size(channelview(image_square), 1) # Number of input channels (e.g., RGB)
    # nout = 64  # Desired output dimensionality for each patch embedding

    # Create a `PatchEmbedLinear` instance
    patch_embed = PatchEmbedLinear{Float64}(img_size, patch_size, nin, nout)

    # Use the color image of Philip the dog (loaded as `image_square`)
    embedded_patches = patch_embed(image_square)

    # Verify output dimensions
    println("Input image shape: ", size(image_square))
    println("Number of patches: ", patch_embed.num_patches)
    println("Embedded patches shape: ", size(embedded_patches))

    # Visualize the embedded patches (first few)
	npatch = min(10, size(embedded_patches, 1))
    heatmap(embedded_patches[1:npatch, :],
        title="First 10 Embedded Patches",
        xlabel="Embedding Dimension",
        ylabel="Patch Index",
        c=:plasma,
        clabel="Value")
end


# ╔═╡ ff7337df-dd2a-4688-9623-abac908491c5
function visualize_patch_embedding(image::Matrix{<:RGB}, patch_size::Int, embedded_patches::Matrix{Float64})
    # Extract patches for visualization
    patches = extract_patches(image, patch_size)
    
    # Calculate grid dimensions for patches
    n_patches = length(patches)
    grid_dim = div(size(image, 1), patch_size)  # Grid dimensions (e.g., 16x16 for 256x256 with patch_size=16)
    
    # Start with the original image
    p1 = plot(image, title="Input Image", size=(800, 800), color=:grays, axis=false)
    
    # Overlay transparent patches
    for i in 0:grid_dim-1
        for j in 0:grid_dim-1
            # Draw rectangles for patches
            rectangle = Shape([j*patch_size, (j+1)*patch_size, (j+1)*patch_size, j*patch_size],
                              [i*patch_size, i*patch_size, (i+1)*patch_size, (i+1)*patch_size])
            plot!(rectangle, lw=1.5, linealpha=0.7, fillalpha=0.0, color=:red, legend=false)
        end
    end
    
    # Visualize embeddings as a heatmap
    p2 = heatmap(embedded_patches, title="Patch Embeddings", xlabel="Embedding Dimension",
                 ylabel="Patch Index", color=:viridis, size=(800, 800))
    
    # Combine the two plots
    plot(p1, p2, layout=(1, 2), size=(1200, 800))
end


# ╔═╡ fa4a03f5-f52a-4fbb-bb51-4f7daca912ac
# Example Usage
let
    img_size = size(image_square, 1)  # Image size (assumes square image)
    nin = size(channelview(image_square), 1)  # Number of input channels (RGB)
    # nout = 64  # Desired embedding dimension
    
    # Initialize PatchEmbedLinear
    patch_embed = PatchEmbedLinear{Float64}(img_size, patch_size, nin, nout)
    
    # Apply PatchEmbedLinear to the image
    embedded_patches = patch_embed(image_square)
    
    # Visualize the process
    visualize_patch_embedding(image_square, patch_size, embedded_patches)
end

# ╔═╡ 8e813069-1265-4469-980d-e1450d6ae173
md"""
## Positional Encoding
One reason why CNNs worked so well for image recognition is because they have an inductive bias for local structure. In a Trasnformer, every token can attend to every other token in the sequence. Because self-attention operation is permutation invariant, it is important to use proper positional encoding to provide order information to the model. The positional encoding $\mathbf{P} \in \mathbb{R}^{L \times d}$ has the same dimension as the input embedding, so it can be added on the input directly. 

---

The vanilla Transformer considered two types of encodings:
- (1) _Sinusoidal positional encoding_: Each dimension of the positional encoding corresponds to a sinusoid of different wavelengths in different dimensions. 

![Sinusoidal Positional Encoding](https://github.com/qsimeon/julia_class_project/blob/main/sine_encoding.jpg?raw=True)
- (2) _Learned positional encoding_: As its name suggests, assignes each element in a sequence with a learned column vector which encodes its absolute position.
---
"""

# ╔═╡ a6fc3703-585d-453f-a30a-25d080ab053d
md"""
We will implement the latter (2) by implementing and `Embedding` module since it is straightforward and becuase embeddig layers are extremely useful and ubiquitous in deep learning code.

![Embedding Table Explanation](https://github.com/qsimeon/julia_class_project/blob/main/how_embedding_works.jpg?raw=true)
---
"""

# ╔═╡ e6bff9ce-2cb0-4974-a2b5-d04243e8f0ba
### 8. Embedding
struct Embedding{T<:Real}
    emb::Matrix{T} # Shape: (num_embeddings, embedding_dim)

    function Embedding{T}(num_embeddings::Int, embedding_dim::Int) where T<:Real
        emb = randn(T, num_embeddings, embedding_dim) # Randomly initialize the embedding matrix
        return new{T}(emb)
    end

    function (embedding::Embedding{T})(indices::Vector{Int}) where T<:Real
        # Ensure indices are valid
        @assert all(1 .≤ indices .≤ size(embedding.emb, 1)) "Indices out of range"
        # Perform lookup for each index
        return embedding.emb[indices, :]
    end
end


# ╔═╡ a87e64c5-e8f4-4e61-8c66-3fe4c22e5c1c
# Test Embedding module
let
    num_embeddings = 100  # Number of patches
    embedding_dim = 64     # Embedding dimension

    # Create the embedding module
    embedding = Embedding{Float64}(num_embeddings, embedding_dim)

    # Sample indices to lookup
    indices = [1, 10, 50, 100]

    # Lookup embeddings for the indices
    embeddings = embedding(indices)

    # Verify output
    println("Indices: ", indices)
    println("Embedding shape: ", size(embeddings))  # Should be (length(indices), embedding_dim)
end


# ╔═╡ 8c355943-964f-4db0-a1ec-dd160b282583
md"""
## Almost there!

We just need to define a few more layers to put together the **ViT**.

---
"""

# ╔═╡ 867dae62-6570-4131-8713-7867196a8736
struct Sequential{T<:Real}
    # This is an array of modules where each module is applied sequentially to the input.
    seq::AbstractVector

    function Sequential{T}(seq::AbstractVector) where T<:Real
        return new{T}(seq)
    end

    function (sequential::Sequential{T})(x::Array{T}) where T<:Real
        for mod in sequential.seq
            x = mod(x)
        end
        return x
    end
end

# ╔═╡ c3fce17e-06eb-4982-bcef-86b8c53f78ef
# Test `Sequential` implementation
let
    dim, mlp_dim, nout = 3, 8, 5
    layer1 = Linear{Float64}(dim, mlp_dim)
    layer2 = Linear{Float64}(mlp_dim, nout)

    # Initialize `Sequential` with two layers
    sequential = Sequential{Float64}([layer1, layer2])

    # Create a sample input
    X = randn(Float64, n_tokens, dim)  # Input with n_tokens tokens of dim-dimensional vectors

    # Apply `Sequential`
    output = sequential(X)

    # Verify output shape
    println("Input shape: ", size(X))          # Should be (10, dim)
    println("Output shape: ", size(output))   # Should be (10, nout)
end

# ╔═╡ 307db93b-20f3-4dd1-9dd7-e05780592245
struct LayerNorm{T<:Real}
    dim::Int

    function LayerNorm{T}(dim::Int) where T<:Real
        return new{T}(dim)
    end

    function (layernorm::LayerNorm{T})(X::Array{T}) where T<:Real
        mean_val = mean(X; dims=layernorm.dim)
        std_val = std(X; dims=layernorm.dim)
        return (X .- mean_val) ./ (std_val .+ eps(T))
    end
end


# ╔═╡ 6e615061-9600-4a98-8c15-c30110dde0ee
struct Parameter{T<:Real}
    param::Vector{T}

    function Parameter{T}(dim::Int) where T<:Real
        param = randn(T, dim)
        return new{T}(param)
    end
end

# ╔═╡ e32c2cb0-2862-4f7a-9470-61ea5544202e
struct VisionTransformer{T<:Real}
    patch_embed::PatchEmbedLinear{T}          # Patch embedding layer
    pos_E::Embedding{T}                       # Positional encoding
    cls_token::Parameter{T}                   # Learned class embedding token
    transformer::Transformer{T}               # Transformer encoder
    head::Sequential{T}                       # Classification head

    function VisionTransformer{T}(
        n_channels::Int, nout::Int, img_size::Int, patch_size::Int, dim::Int,
        attn_dim::Int, mlp_dim::Int, num_heads::Int, num_layers::Int
    ) where T<:Real
        # Initialize each component
        patch_embed = PatchEmbedLinear{T}(img_size, patch_size, n_channels, dim)
        pos_E = Embedding{T}((img_size ÷ patch_size)^2, dim)
        cls_token = Parameter{T}(dim)
        transformer = Transformer{T}(dim, attn_dim, mlp_dim, num_heads, num_layers)
        head = Sequential{T}([LayerNorm{T}(dim), Linear{T}(dim, nout)])  # LayerNorm along embedding dim

        return new{T}(patch_embed, pos_E, cls_token, transformer, head)
    end

    function (vt::VisionTransformer{T})(img::Matrix{<:RGB}; return_attn::Bool=false) where T<:Real
        # Generate patch embeddings
        embs = vt.patch_embed(img)  # Shape: (num_patches, dim)

        # Add positional encoding
        N, D = size(embs)  # Number of patches (N) and embedding dimension (D)
        pos_ids = collect(1:N)  # Generate positional indices
        embs .+= vt.pos_E(pos_ids)  # Add positional encodings to embeddings

        # Add the class token
        cls_token = vt.cls_token.param  # Shape: (dim,)
        x = vcat(cls_token', embs)  # Shape: (N+1, dim)

        # Apply the transformer
        x, alphas = vt.transformer(x; attn_mask=nothing, return_attn=return_attn)

        # Pass through the classification head
        cls_token_out = reshape(x[1, :], 1, :)  # Reshape into a matrix for `Sequential`
        out = vt.head(cls_token_out)[1, :]  # Final output as a vector

        return out, alphas
    end
end


# ╔═╡ 2f5badaf-4342-42ec-8240-c5c642c1fa8f
# Test Vision Transformer implementation for single-image inputs
let
    # Define hyperparameters
    img_size = 256           # Image size (assumes square image)
    # patch_size = 16          # Patch size
    n_channels = 3           # Number of input channels (RGB)
    dim = 64                 # Embedding dimension
    attn_dim = 128           # Attention hidden dimension
    mlp_dim = 256            # Feed-forward network hidden dimension
    num_heads = 4            # Number of attention heads
    num_layers = 6           # Number of transformer layers
    # nout = 10                # Number of output classes (e.g., for classification)

    # Initialize the Vision Transformer
    vt = VisionTransformer{Float64}(
        n_channels, nout, img_size, patch_size, dim, attn_dim, mlp_dim, num_heads, num_layers
    )

    # Use the color image of Philip the dog (loaded as `image_square`)
    out, alphas = vt(image_square, return_attn=true)

    # Verify output dimensions
    println("Output shape: ", size(out))  # Should be (nout,)
    println("Attention weights shape: ", size(alphas))  # Should be (num_layers, num_heads, N+1, N+1)

    # Visualize one attention map (optional)
    heatmap(
        alphas[1][1, :, :],  # Visualize the attention weights for the first layer and first head
        title="Attention Map",
        xlabel="Token Index",
        ylabel="Token Index",
        c=:viridis,
        clabel="Attention Weight (first layer)"
    )
end


# ╔═╡ 33a7fb9e-838d-4b5b-9310-5d92719d7eaf
md"""
## Loading an image dataset (`CIFAR10`)
"""

# ╔═╡ 22689f54-30d2-41fc-89ca-5bf0f95e855d
begin
	# Load CIFAR-10 training data
	train_dataset = CIFAR10(dir="cifar/", split=:train)
	# test_dataset = CIFAR10(dir="cifar/", split=:test)

	subset_size = 1000  
	train_subset = train_dataset.features[:, :, :, 1:subset_size]
	train_labels = train_dataset.targets[1:subset_size]
end

# ╔═╡ 1831af2d-587f-40ed-80bc-dd96595aaccf
let 
	# Extract a single CIFAR-10 image and label
    img_data = train_dataset.features[:, :, :, 1]  # Shape: 32x32x3 (HWC format)
    img_data_permuted = permutedims(img_data, (3, 1, 2))  # CHW format for the VisionTransformer
    rgb_image = colorview(RGB, img_data_permuted)
    target_label = train_dataset.targets[1]  # Class label (1-based index)

    # Display the CIFAR-10 image
	rgb_image
end

# ╔═╡ f708229e-d2a2-424c-91f0-3bffda23fe53
md"""
## Cross-Entropy Loss Function
We will define the cross-entropy loss function. For multi-class classification with $C$ classes, the cross-entropy loss for a single sample is given by:

```math
\text { Loss }=-\sum_{c=1}^C y_c \log \left(\hat{y}_c\right)
```

Here:
- ``y_c`` is 1 if the sample belongs to class $c$, otherwise 0.
- ``\hat{y}_c`` is the predicted probability for class $c$.
"""

# ╔═╡ 2a5da94c-dd22-450f-93b9-3e8298308488
function cross_entropy_loss(predictions::Matrix{Float64}, targets::Vector{Int})
    # Convert targets to one-hot encoding
    num_samples, num_classes = size(predictions)
    one_hot_targets = zeros(Float64, num_samples, num_classes)
    for i in 1:num_samples
        one_hot_targets[i, targets[i]] = 1.0
    end
	# Compute log of softmax predictions
    log_probs = log.(softmax(predictions, dims=2))
    # Compute the loss
	loss = -sum(one_hot_targets .* log_probs) / num_samples
    return loss
end

# ╔═╡ a2418e79-b2a3-4310-ba6e-7b0af50264ff
# Compute gradient for the VisionTransformer
function compute_gradient(vit_model::VisionTransformer{T}, img::Matrix{<:RGB}, target::Int) where T<:Real
    function loss_fn()
        # Forward pass through the model
        output, _ = vit_model(img)
        # Compute the loss (batch size = 1)
        y_pred = reshape(output, 1, :)  # Ensure output is a matrix
        return cross_entropy_loss(y_pred, [target])
    end

    # Compute gradients of the loss function with respect to model parameters
    grad = Enzyme.gradient(loss_fn, vit_model)
    return grad
end

# ╔═╡ e6f6f744-7179-4d45-94fa-de0b3bc303bf
# Test the VisionTransformer with gradient computation
let
    # Initialize the Vision Transformer
    img_size = 32          # CIFAR-10 image size
    patch_size = 4         # Patch size
    n_channels = 3         # Number of input channels (RGB)
    dim = 64               # Embedding dimension
    attn_dim = 128         # Attention hidden dimension
    mlp_dim = 256          # Feed-forward network hidden dimension
    num_heads = 4          # Number of attention heads
    num_layers = 6         # Number of transformer layers
    nout = 10              # Number of output classes (CIFAR-10 has 10 classes)

    vit_model = VisionTransformer{Float64}(
        n_channels, nout, img_size, patch_size, dim, attn_dim, mlp_dim, num_heads, num_layers
    )

    # Compute gradients
    grad = compute_gradient(vit_model, img_data_permuted, target_label)

    # Print gradient details
    println("Gradient: ", grad)
end


# ╔═╡ 9cf11cbd-4e1e-4a01-be53-212b53a7bc25


# ╔═╡ Cell order:
# ╠═4cc97f4d-7a4c-487a-8684-1edd1bb963a5
# ╠═ddf6ac4d-df08-4e73-bc60-4925aa4b94c8
# ╠═970f0e2b-459b-4baa-ae30-886c2bada7b4
# ╠═191c435c-4094-4326-9e18-0ee8dc3058ab
# ╠═2348f0c3-5fc1-424f-8a56-c00c52ca9a4f
# ╠═afe50e6c-9e61-4246-a8ac-bebc83e2715c
# ╠═ddc663b2-9de3-11ef-1d3a-9f172c4dda5f
# ╠═9adfff6a-e83e-4266-8bae-67f4a16e011f
# ╠═5a498179-0be9-4e70-988f-14575d12a396
# ╠═c3eaadcf-a06d-4469-ba9a-399043e72a9f
# ╠═245ce308-8fc2-4b31-8aa6-d7c1d33b61ca
# ╠═1c2692a1-e8a0-4926-ad42-3787671eeb51
# ╠═74eb85f0-ea48-48cd-b732-4d97f4883c85
# ╠═c8d32f75-83a3-40d7-b136-4bf5966612a0
# ╠═e9ce397a-b315-470d-9534-cdd1dad12a1b
# ╠═661d546e-4182-4199-9b2d-3c5eb90bc07f
# ╠═03759beb-1ac3-4bd7-800b-f4461edb58b1
# ╠═c12ffac7-7533-42fa-a721-30938396e898
# ╠═a7f306c1-12aa-4ecc-9764-70a72c41bd67
# ╠═2f40b3cf-e690-40be-8e5c-b66e022c505d
# ╠═97c1b967-b634-4ff0-8007-939bf8ea87fa
# ╠═9bd646c3-ef9d-4b06-a598-267c0cbdff4a
# ╠═e88482de-8685-47d7-9cbb-78328eed8244
# ╠═da2f6c55-17c5-495c-8ba3-5b2dc50a17f1
# ╠═fd2a796f-b683-4792-a976-b4071fda58a0
# ╠═7eb8e4ec-80ae-4744-b21d-8b36885ff98c
# ╠═90019339-1fdf-4541-b71b-a00b9ef7d904
# ╠═db19fde5-434c-4030-9a61-0200ddca659f
# ╠═a2bf29b4-174d-42e1-94b6-39822556349c
# ╠═ffeafe79-65b5-4c75-aaf1-e83bc8ca17cc
# ╠═c3d2a61c-2caa-4f52-acab-8a0b89e5aac5
# ╠═9e705315-d646-4373-854d-47a9f9d9076b
# ╠═5a0607bc-bf03-4b19-894f-1bcfd68a0762
# ╠═b35b28dd-e992-49a0-8e01-abd3e26ad093
# ╠═8a0abb17-b7f0-4952-b5c1-0d52095cf2bf
# ╠═44f39ba0-68e6-450d-a7fa-99f180a48b67
# ╠═a2ff04a3-4118-47b8-b768-fc2a4986167b
# ╠═9d6cd065-5f25-4943-b155-3602db474bff
# ╠═02fb8ff3-647e-4d55-8c2b-a1d9066338ed
# ╠═ff7337df-dd2a-4688-9623-abac908491c5
# ╠═fa4a03f5-f52a-4fbb-bb51-4f7daca912ac
# ╠═8e813069-1265-4469-980d-e1450d6ae173
# ╠═a6fc3703-585d-453f-a30a-25d080ab053d
# ╠═e6bff9ce-2cb0-4974-a2b5-d04243e8f0ba
# ╠═a87e64c5-e8f4-4e61-8c66-3fe4c22e5c1c
# ╠═8c355943-964f-4db0-a1ec-dd160b282583
# ╠═867dae62-6570-4131-8713-7867196a8736
# ╠═c3fce17e-06eb-4982-bcef-86b8c53f78ef
# ╠═307db93b-20f3-4dd1-9dd7-e05780592245
# ╠═6e615061-9600-4a98-8c15-c30110dde0ee
# ╠═e32c2cb0-2862-4f7a-9470-61ea5544202e
# ╠═2f5badaf-4342-42ec-8240-c5c642c1fa8f
# ╠═33a7fb9e-838d-4b5b-9310-5d92719d7eaf
# ╠═22689f54-30d2-41fc-89ca-5bf0f95e855d
# ╠═1831af2d-587f-40ed-80bc-dd96595aaccf
# ╠═f708229e-d2a2-424c-91f0-3bffda23fe53
# ╠═2a5da94c-dd22-450f-93b9-3e8298308488
# ╠═a2418e79-b2a3-4310-ba6e-7b0af50264ff
# ╠═e6f6f744-7179-4d45-94fa-de0b3bc303bf
# ╠═9cf11cbd-4e1e-4a01-be53-212b53a7bc25
