using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        public readonly Neuron[][] Neurons;     // Layers of neurons
        public IActivator[] Activators;          // Activators for each layer

        public Network(IActivator activator, params int[] neuronsInEachLayer)   // including the input layer
        {
            Neurons = neuronsInEachLayer
                .Skip(1)                          // Skip the input layer
                .Select((count, layer) =>
                    Enumerable.Range(0, count)
                        .Select(index => new Neuron(this, layer, index, neuronsInEachLayer[layer]))
                        .ToArray())
                .ToArray();

            Activators = Enumerable
                .Repeat(activator, neuronsInEachLayer.Length - 1)
                .ToArray();
        }
    }
}
