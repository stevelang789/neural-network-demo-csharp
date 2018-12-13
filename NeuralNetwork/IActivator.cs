namespace NeuralNetwork
{
    public interface IActivator
    {
        void ComputeOutputs(FiringNeuron[] layer);
        double GetActivationSlopeAt(FiringNeuron neuron);
    }
}
