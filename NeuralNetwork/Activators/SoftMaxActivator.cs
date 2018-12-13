using System;

namespace NeuralNetwork.Activators
{
    public class SoftMaxActivator : IActivator
    {
        public void ComputeOutputs(FiringNeuron[] layer)
        {
            double sum = 0;

            foreach (var neuron in layer)
            {
                neuron.Output = Math.Exp(neuron.TotalInput);
                sum += neuron.Output;
            }

            foreach (var neuron in layer)
            {
                var oldOutput = neuron.Output;
                neuron.Output = neuron.Output / (sum == 0 ? 1 : sum);
            }
        }

        public double GetActivationSlopeAt(FiringNeuron neuron)
        {
            double y = neuron.Output;
            return y * (1 - y);
        }
    }
}
