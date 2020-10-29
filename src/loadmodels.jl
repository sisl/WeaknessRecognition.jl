using CUDA
using FailureRepresentation
using FailureRepresentation.SystemUnderTest
using FailureRepresentation.Autoencoder
using Flux

sut = SystemUnderTest.load("models/sut.bson")
autoencoder = Autoencoder.load("models/autoencoder.bson")
