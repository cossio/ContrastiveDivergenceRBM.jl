import Aqua
import ContrastiveDivergenceRBM
using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(ContrastiveDivergenceRBM; ambiguities = false)
end
