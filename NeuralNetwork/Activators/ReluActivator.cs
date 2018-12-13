namespace NeuralNetwork.Activators
{
    public class ReluActivator : IActivator
    {
        public void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = neuron.TotalInput > 0 ? neuron.TotalInput : neuron.TotalInput / 100;
        }

        public double GetActivationSlopeAt(FiringNeuron neuron) => neuron.TotalInput > 0 ? 1 : .01;
    }
}
