"""
    cd!(rbm, data)

Trains the RBM on data using Contrastive divergence.
"""
function cd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    iters::Int = 1, # number of gradient updates
    steps::Int = 1, # MC steps to update fantasy chains
    optim::AbstractRule = Adam(), # optimizer rule
    moments = moments_from_samples(rbm.visible, data), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    rescale::Bool = true, # normalize weights to unit norm (for continuous hidden units only)

    callback = Returns(nothing), # called for every batch

    shuffle::Bool = true,

    # parameters to optimize
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps) # initialize optimiser state
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd,)) in zip(1:iters, infinite_minibatches(data; batchsize, shuffle))
        # fantasy chains, sampled from the data
        vm = sample_v_from_v(rbm, vd; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # reset gauge
        rescale && rescale_weights!(rbm)
        zerosum && zerosum!(rbm)

        callback(; rbm, optim, iter, vm, vd, ∂)
    end
    return state, ps
end
