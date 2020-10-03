using MLDatasets
using Flux
using DataFrames
using Plots
using RDatasets
# Quick Example to introduce Params(): Linear Regression
# random initial parameters
β = rand(2,1)
α = rand(1)

f̂(x) = x * β .+ α 
loss(x,y) = sum((f̂(x) - y).^2)

x = rand(10,2)
y = rand(10)

loss(x,y) 

Δ = gradient(() -> loss(x, y), params(β, α))
β̂ = Δ[β]
α̂ = Δ[α]

β .-= 0.01 .* β̂
α .-= 0.01 .* α̂

loss(x, y)


## Neural Network with Autodiff
iris = dataset("datasets", "iris")
y = DataFrames.categorical(iris["Species"])
y =  map(y -> Int(y), CategoricalArrays.order(y.pool)[y.refs])

W_1 = randn(4,25)
b_1 = randn(1)
W_2 = randn(25,3)
b_2 = randn(1)

function f(x)
    layer_1 = tanh.(x' * W_1 .+ b_1)
    out_layer = softmax((layer_1 * W_2 .+ b_2)')
    return out_layer
end

function loss(x, y)
    -log(f(x)[y])
end
total_loss = 0
for i ∈ 1:(nrow(iris))
    local x = Array(iris[i:4,1:4])'
    local y_act = y[i]

    Δ = gradient(() -> loss(x, y_act), params(W_1,W_2,b_1,b_2))
    
    Ŵ_1 = Δ[W_1]
    Ŵ_2 = Δ[W_2]
    b̂_1 = Δ[b_1]
    b̂_2 = Δ[b_2]

    W_1 .-= 0.01 .* Ŵ_1
    W_2 .-= 0.01 .* Ŵ_2
    b_1 .-= 0.01 .* b̂_1
    b_2 .-= 0.01 .* b̂_2

    global total_loss += loss(x, y_act)

end
println(total_loss)


