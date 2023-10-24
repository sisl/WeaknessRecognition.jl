using CUDA
using WeaknessRecognition
using WeaknessRecognition.SystemUnderTest
using WeaknessRecognition.Autoencoder
using Flux

sut = SystemUnderTest.load("models/sut.bson")
autoencoder = Autoencoder.load("models/autoencoder.bson")
encoder = Autoencoder.load("models/encoder.bson", :encoder)
