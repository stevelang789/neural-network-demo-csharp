using System;

namespace NeuralNetwork.Activators
{
    public class LogisticSigmoidActivator : IActivator
    {
        public void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = 1 / (1 + Math.Exp(-neuron.TotalInput));
        }

        public double GetActivationSlopeAt(FiringNeuron neuron)
            => neuron.Output * (1 - neuron.Output);
    }
}
