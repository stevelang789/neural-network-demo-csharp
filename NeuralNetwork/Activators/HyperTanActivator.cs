using System;

namespace NeuralNetwork.Activators
{
    public class HyperTanActivator : IActivator
    {
        public void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = Math.Tanh(neuron.TotalInput);
        }

        public double GetActivationSlopeAt(FiringNeuron neuron)
        {
            var tanh = neuron.Output;
            return 1 - tanh * tanh;
        }
    }
}
