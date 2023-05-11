using Test: @test
using LinearAlgebra: svd
using RestrictedBoltzmannMachines: RBM, Spin
using Optimisers: Adam
using ContrastiveDivergenceRBM: cd!

N = 500
M = 300
K = 5

rbm = RBM(Spin((N,)), Spin((M,)), randn(N, M) / √N)
v = sign.(randn(N, K))
train_data = repeat(v, 1, 100)

niter = 1000
save_sv_every = 100

_myiter = 0
sv_history = zeros(length(rbm.hidden), niter ÷ save_sv_every)
function callback(; rbm, kw...)
    rbm.visible.θ .= 0
    rbm.hidden.θ .= 0
    global _myiter += 1
    if _myiter % save_sv_every == 0
        sv_history[:, _myiter ÷ save_sv_every] .= svd(rbm.w).S
    end
end

cd!(rbm, train_data; iters=niter, batchsize=128, callback, l2_weights=0.001, steps=10, optim=Adam(0.0001))
