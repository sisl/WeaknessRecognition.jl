module FailureRepresentation

include("SystemUnderTest.jl")
using .SystemUnderTest
export load, predict, getdata


include("Autoencoder.jl")
using .Autoencoder
export load, autoencoder

end # module
