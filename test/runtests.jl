using Test
using StreamTensor

@testset "StreamTensor.jl" begin
    include("test_index.jl")
    include("test_tensor.jl")
    include("test_contraction.jl")
    include("test_combiner.jl")
    include("test_deomposition.jl")
    include("test_mps.jl")
    include("test_apply.jl")
end
