module ContrastiveDivergenceRBM

using Optimisers: AbstractRule, setup, update!, Adam
using RestrictedBoltzmannMachines: RBM, moments_from_samples, sample_from_inputs,
    zerosum!, rescale_weights!, infinite_minibatches, sample_v_from_v,
    ∂free_energy, ∂regularize!

include("cd.jl")

end # module
