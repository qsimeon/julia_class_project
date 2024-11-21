### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ddf6ac4d-df08-4e73-bc60-4925aa4b94c8
md"""
### Our project is to implement a simple Vision Transformer in Julia!

The Transformer architecture, introduced in the paper _Attention Is All You Need_ (Vaswani et al., 2017), is the most ubiquitous neural network architecture in modern machine learning. Its parallelism and scalability to large problems has seen it adopted in domains beyong those it was traditionally considered for (sequential data) and it quickly replaced convolutional neural networks for image-based tasks. 
"""

# ╔═╡ 191c435c-4094-4326-9e18-0ee8dc3058ab
md"""
![ViT Model](https://github.com/qsimeon/julia_class_project/blob/e698587c2c2b7455404e6126c06f4ec04c463032/vit_arch.jpg?raw=true)
"""

# ╔═╡ 2348f0c3-5fc1-424f-8a56-c00c52ca9a4f
md"""
Let’s start by defining key components of a Vision Transformer (ViT) model using Julia structs and parametric types, similar to the structure we implemented in Homework 3. We will implement the `AttentionHead`, `MultiHeadedAttention`, and `FeedForwardNetwork` layers as Julia structs. This will set up the parts which get combined together in the `Transformer` model.
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

    function (head::AttentionHead{T})(X::Matrix{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
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

    function (mha::MultiHeadedAttention{T})(X::Matrix{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
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

# ╔═╡ 245ce308-8fc2-4b31-8aa6-d7c1d33b61ca
# with mask
let
	dim, attn_dim = 3, n_tokens
	head = AttentionHead{Float64}(dim, attn_dim)
	X = randn(Float64, n_tokens, dim)  # example 3-D input of n_tokens

	p1 = heatmap(X, title="Input", aspect_ratio=1, colorbar=false)

	Q = randn(attn_dim, dim)
	K = randn(attn_dim, dim)
	V = randn(dim, dim)
	p2 = heatmap(Q, title="Q", aspect_ratio=1, colorbar=false)
	p3 = heatmap(K, title="K", aspect_ratio=1, colorbar=false)
	p4 = heatmap(V, title="V", aspect_ratio=1, colorbar=false)
	p5 = heatmap(X * transpose(Q), title="XQ", aspect_ratio=1, colorbar=false)
	p6 = heatmap(X * transpose(K), title="XK", aspect_ratio=1, colorbar=false)
	p7 = heatmap(X * transpose(V), title="XV", aspect_ratio=1, colorbar=false)
	scores = Q * transpose(K) / sqrt(head.n_hidden)
	p8 = heatmap(scores, title="attn", aspect_ratio=1, colorbar=false)

	mask = UpperTriangular(ones(n_tokens, n_tokens))
	p85 = heatmap(mask, title="mask", aspect_ratio=1, colorbar=false)
	scores_mask = scores .* mask .+ (1 .- mask) 
	p9 = heatmap(scores_mask, title="masked attn", aspect_ratio=1, colorbar=false)

	alpha = softmax(scores_mask, dims=ndims(scores_mask)) 
	p10 = heatmap(alpha, title="alpha", aspect_ratio=1, colorbar=false)

	attn_output = alpha * (X * transpose(V))
	p11 = heatmap(attn_output, title="output", aspect_ratio=1, colorbar=false)

	plot!([p1,p2,p3,p4,p5,p6,p7,p8,p85,p9,p10,p11]..., layout=(3,4))
end

# ╔═╡ c12ffac7-7533-42fa-a721-30938396e898
### 3. Feed-Forward Network (FFN)
struct FeedForwardNetwork{T<:Real}
    W1::Matrix{T} # Shape: (n_hidden, dim)
    W2::Matrix{T} # Shape: (dim, n_hidden)
    b1::Vector{T} # Shape: (n_hidden,)
    b2::Vector{T} # Shape: (dim,)

    function FeedForwardNetwork{T}(dim::Int, n_hidden::Int) where T<:Real
		# Our FFN outputs tokens with the same dimension as the input tokens
        return new{T}(randn(T, n_hidden, dim), randn(T, dim, n_hidden), randn(T, n_hidden), randn(T, dim))
    end

    function (ffn::FeedForwardNetwork{T})(X::Matrix{T}) where {T<:Real}
        # X is expected to be an input token matrix with shape (N, dim)
        X = X * transpose(ffn.W1) .+ ffn.b1'  # Shape: (N, n_hidden)
        X = max.(0, X)  # ReLU activation
        return X * transpose(ffn.W2) .+ ffn.b2'  # Shape: (N, dim)
    end
end


# ╔═╡ a7f306c1-12aa-4ecc-9764-70a72c41bd67
# Test `FeedForwardNetwork` implementation
let
	dim, mlp_dim = 3, 8
	ffn = FeedForwardNetwork{Float64}(dim, mlp_dim)
	X = randn(Float64, n_tokens, dim)  # example 3-D input of n_tokens
	ffn_output = ffn(X)
	println("feedforward output shape: ", size(ffn_output))
end

# ╔═╡ 2f40b3cf-e690-40be-8e5c-b66e022c505d
md"""
### Recap so far

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
### 4. Attention Residual
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
	function (residual::AttentionResidual{T})(X::Matrix{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
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
### 5. Attention Residual
struct Transformer{T<:Real}
    layers::Vector{AttentionResidual{T}}  # Sequence of AttentionResidual blocks

	# Constructor: initializes a sequence of attention residual blocks
    function Transformer{T}(dim::Int, attn_dim::Int, mlp_dim::Int, num_heads::Int, num_layers::Int) where T<:Real
        layers = [AttentionResidual{T}(dim, attn_dim, mlp_dim, num_heads) for _ in 1:num_layers]
        return new{T}(layers)
    end

	# Apply the Transformer model to input X
    function (transformer::Transformer{T})(X::Matrix{T}, attn_mask::Union{Nothing, Matrix{T}}=nothing) where {T<:Real}
        collected_alphas = []  # To store attention weights from each layer
        for layer in transformer.layers
            X, alphas = layer(X, attn_mask)  # Apply each residual block
            push!(collected_alphas, alphas)  # Collect attention weights
        end
		# Return the final output and collected attention weights from all layers
        return X, collected_alphas
    end
end


# ╔═╡ e88482de-8685-47d7-9cbb-78328eed8244
md"""
### Testing the AttentionResidual and Transformer

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
	output, collected_alphas = transformer(X)
	println("Transformer output shape: ", size(output))
	println("Collected attention weights shape: ", size(collected_alphas[1]), " for ", num_layers, " layers")
end

# ╔═╡ 7eb8e4ec-80ae-4744-b21d-8b36885ff98c
md"""
#### Our modules so far build up the Transfomer

- **AttentionHead**: Implements a single attention head, creating query, key, and value projections, computing the attention scores, and applying a softmax.
- **MultiHeadedAttention**: Combines multiple `AttentionHead`s, concatenates their outputs, and applies a final linear transformation.
- **FeedForwardNetwork**: A simple feed-forward network with two linear layers and a ReLU activation in between.
- **AttentionResidual**: Combines multi-head attention and feed-forward network layers with residual connections.
- **Transformer**: Stacks multiple `AttentionResidual` layers to form the complete Transformer encoder.

---

We've set up a basic structure of a Transformer using callable structs, parametric types, and matrix operations. 

Let's view our Julia callable struct implemenation side-by-side with a bare-bones implementation in PyTorch.

**TODO:** Side-by-side comparison.
"""

# ╔═╡ db19fde5-434c-4030-9a61-0200ddca659f
md"""
We want to make a Vision Transformer. This requires some additional layers for image processing: patch embedding and positional encoding. 

**TODO:** Implement `PatchEmbed` in Julia and make a visual example of applying it to some image like this:

![Patch Image](https://github.com/qsimeon/julia_class_project/blob/main/patch_embed.jpg?raw=true)
"""

# ╔═╡ a2bf29b4-174d-42e1-94b6-39822556349c
md"""
It turns out the patch embedding is can be implemented by applying a strided convolution. However, we will take the more direct and visualizable approach of chopping up an image into patches and linearly projecting the vector that is the flattened patch to the desired dimensionality. 

Remember Transformers operate on tokens i.e. transformations of tokens. What we are doing here is essentially *tokenizing* our image data.
"""

# ╔═╡ a2ff04a3-4118-47b8-b768-fc2a4986167b
### 5. PatchEmbedLinear struct 
begin
	struct PatchEmbedLinear{T<:Real}
	    img_size::Int
	    patch_size::Int
	    nin::Int
	    nout::Int
	    num_patches::Int
	    W::Matrix{T} # Linear projection weights
	
	    function PatchEmbedLinear{T}(img_size::Int, patch_size::Int, nin::Int, nout::Int) where T<:Real
	        @assert img_size % patch_size == 0 "img_size must be divisible by patch_size"
	        
	        num_patches = (img_size ÷ patch_size)^2
	        W = randn(T, nout, patch_size^2 * nin) # Linear projection matrix for each patch
	        return new{T}(img_size, patch_size, nin, nout, num_patches, W)
	    end
	end
end


# ╔═╡ 33d476fa-4185-462d-94d3-b7011bb52d36
function conv2d(input::AbstractArray, kernel::AbstractArray, stride::Int)
    # Input shape: (Height, Width, Channels)
    # Kernel shape: (KernelHeight, KernelWidth, Channels, OutputChannels)
    # Stride is an integer value

    h_in, w_in, c_in = size(input)
    k_h, k_w, _, c_out = size(kernel)

    # Calculate the output dimensions based on input size and stride
    h_out = floor(Int, (h_in - k_h) / stride) + 1
    w_out = floor(Int, (w_in - k_w) / stride) + 1

    # Initialize the output array
    output = zeros(Float64, h_out, w_out, c_out)

    for i in 1:h_out
        for j in 1:w_out
            for k in 1:c_out
                # Apply the convolution to the region of interest
                for di in 1:k_h
                    for dj in 1:k_w
                        for d in 1:c_in
                            output[i, j, k] += input[(i-1)*stride + di, (j-1)*stride + dj, d] * kernel[di, dj, d, k]
                        end
                    end
                end
            end
        end
    end

    return output
end


# ╔═╡ 8b9eb4c6-b9bc-4ee3-bb7e-e61f396a12fd
function gaussian_kernel(size::Int, sigma::Float64=1.0)
    @assert isodd(size) "Kernel size must be odd"
    center = (size - 1) / 2
    kernel = zeros(Float64, size, size)
    for i in 1:size
        for j in 1:size
            x, y = i - center - 1, j - center - 1
            kernel[i, j] = exp(- (x^2 + y^2) / (2 * sigma^2))
        end
    end
    return kernel / sum(kernel) # Normalize
end

# ╔═╡ 853e38ee-6526-46c8-8af1-fcfce02fe4f3
# PatchEmbedConv struct
struct PatchEmbedConv
	    img_size::Int  # The width and height of the image (square image)
	    patch_size::Int  # The width of each square patch
	    nin::Int  # The number of input channels
	    nout::Int  # The number of output channels
	
	    function PatchEmbedConv(img_size::Int, patch_size::Int, nin::Int, nout::Int)
	        return new(img_size, patch_size, nin, nout)
	    end
	
		function (p::PatchEmbedConv)(x::AbstractArray)
		    # x should have shape (Height, Width, Channels) for a single image
			x = permutedims(x, (3, 2, 1))
		    kernel = rand(Float64, p.patch_size, p.patch_size, p.nin, p.nout)  # Random kernel
		    stride = p.patch_size
		
		    # Apply conv2d to extract patches
		    patches = conv2d(x, kernel, stride)
		
		    # Flatten the patches into a (num_patches, nout) form
		    num_patches = size(patches, 1) * size(patches, 2)
		    patches_flattened = reshape(patches, num_patches, p.nout)
		
		    return patches_flattened
		end
	end

# ╔═╡ 9e705315-d646-4373-854d-47a9f9d9076b
# Load and preprocess the image
begin
    image_url = "https://github.com/qsimeon/julia_class_project/blob/e698587c2c2b7455404e6126c06f4ec04c463032/reduced_phil.png?raw=true"
    image_file = download(image_url)
    image = load(image_file)

    # Resize the image to a square (e.g., 256x256)
    img_size = 256
    image_square = imresize(image, (img_size, img_size))
end

# ╔═╡ ffeafe79-65b5-4c75-aaf1-e83bc8ca17cc
# Define function to extract patches
function extract_patches(image, patch_size)
    patches = []
    for i in 1:patch_size:size(image, 1)
        for j in 1:patch_size:size(image, 2)
            push!(patches, view(image, i:i+patch_size-1, j:j+patch_size-1))
        end
    end
    return patches
end

# ╔═╡ 5e2a6e26-e0b6-4a7f-a02e-9a2f6708605b
function visualize_patches(image_square, patch_size)
	# Extract patches from image
	patches = extract_patches(image_square, patch_size)
	
	# Calculate grid dimensions (assuming a square grid)
	n_patches = length(patches)
	grid_dim = Int(sqrt(n_patches))
	
	# Initialize a list to hold the individual patch plots
	plot_list = []
	
	# Generate the grid of patches
	for idx in 1:n_patches
		# Create a heatmap for each patch without axis or colorbar
		p = heatmap(patches[idx], color=:grays, axis=false, colorbar=false)
		push!(plot_list, p)
	end
	
	# Display the grid of patches
	plot(plot_list..., layout=(grid_dim, grid_dim), #title="Patches of Image", 
	margin=5mm,size=(800, 800))
end


# ╔═╡ 8a0abb17-b7f0-4952-b5c1-0d52095cf2bf
# Example usage (assuming `image_square` and `extract_patches` are defined):
visualize_patches(image_square, 64)

# ╔═╡ 5257cfb8-ce20-4b73-bc4c-2182a0bd29ad
# key / query demonstration

let
    url = "https://github.com/qsimeon/julia_class_project/blob/main/attention_map.png?raw=true"
    imf = download(url)
    im = load(imf)

    # Resize the image to a square (e.g., 256x256)
    # img_size = 256
    # image_square = imresize(image, (img_size, img_size))
end

# ╔═╡ 1d4a8cd0-fb5f-4e92-a866-e0735a309f54
let
	p = PatchEmbedConv(256, 32, 3, 8)
	patches = p(channelview(image_square))
	size(patches)
end

# ╔═╡ e6bff9ce-2cb0-4974-a2b5-d04243e8f0ba


# ╔═╡ fb8680b6-a78f-4d05-bc60-c738bef2b6c4
using Enzyme

# ╔═╡ 4cc97f4d-7a4c-487a-8684-1edd1bb963a5
using LinearAlgebra, Random, Plots, PlutoUI, Images, MLDatasets, Enzyme

# ╔═╡ Cell order:
# ╠═4cc97f4d-7a4c-487a-8684-1edd1bb963a5
# ╠═ddf6ac4d-df08-4e73-bc60-4925aa4b94c8
# ╠═191c435c-4094-4326-9e18-0ee8dc3058ab
# ╠═2348f0c3-5fc1-424f-8a56-c00c52ca9a4f
# ╠═afe50e6c-9e61-4246-a8ac-bebc83e2715c
# ╠═ddc663b2-9de3-11ef-1d3a-9f172c4dda5f
# ╠═9adfff6a-e83e-4266-8bae-67f4a16e011f
# ╠═5a498179-0be9-4e70-988f-14575d12a396
# ╠═74eb85f0-ea48-48cd-b732-4d97f4883c85
# ╠═c8d32f75-83a3-40d7-b136-4bf5966612a0
# ╠═245ce308-8fc2-4b31-8aa6-d7c1d33b61ca
# ╠═c12ffac7-7533-42fa-a721-30938396e898
# ╠═a7f306c1-12aa-4ecc-9764-70a72c41bd67
# ╠═2f40b3cf-e690-40be-8e5c-b66e022c505d
# ╠═97c1b967-b634-4ff0-8007-939bf8ea87fa
# ╠═9bd646c3-ef9d-4b06-a598-267c0cbdff4a
# ╠═e88482de-8685-47d7-9cbb-78328eed8244
# ╠═da2f6c55-17c5-495c-8ba3-5b2dc50a17f1
# ╠═fd2a796f-b683-4792-a976-b4071fda58a0
# ╠═7eb8e4ec-80ae-4744-b21d-8b36885ff98c
# ╠═db19fde5-434c-4030-9a61-0200ddca659f
# ╠═a2bf29b4-174d-42e1-94b6-39822556349c
# ╠═a2ff04a3-4118-47b8-b768-fc2a4986167b
# ╠═33d476fa-4185-462d-94d3-b7011bb52d36
# ╠═8b9eb4c6-b9bc-4ee3-bb7e-e61f396a12fd
# ╠═853e38ee-6526-46c8-8af1-fcfce02fe4f3
# ╠═9e705315-d646-4373-854d-47a9f9d9076b
# ╠═ffeafe79-65b5-4c75-aaf1-e83bc8ca17cc
# ╠═5e2a6e26-e0b6-4a7f-a02e-9a2f6708605b
# ╠═8a0abb17-b7f0-4952-b5c1-0d52095cf2bf
# ╠═5257cfb8-ce20-4b73-bc4c-2182a0bd29ad
# ╠═1d4a8cd0-fb5f-4e92-a866-e0735a309f54
# ╠═fb8680b6-a78f-4d05-bc60-c738bef2b6c4
# ╠═e6bff9ce-2cb0-4974-a2b5-d04243e8f0ba
