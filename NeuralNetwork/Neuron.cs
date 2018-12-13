using System;
using System.Linq;

namespace NeuralNetwork
{
    public class Neuron
    {
        public readonly Network Net;
        public readonly int Layer, Index;
        public double[] InputWeights;
        public double Bias;

        private static readonly Random Random = new Random();

        public Neuron(Network net, int layer, int index, int inputWeightCount)
        {
            Net = net;
            Layer = layer;
            Index = index;

            Bias = GetSmallRandomNumber();
            InputWeights = Enumerable.Range(0, inputWeightCount).Select(_ => GetSmallRandomNumber()).ToArray();
        }

        public IActivator Activator => Net.Activators[Layer];

        public bool IsOutputNeuron => Layer == Net.Neurons.Length - 1;

        private static double GetSmallRandomNumber() =>
            (.0009 * Random.NextDouble() + .0001) * (Random.Next(2) == 0 ? -1 : 1);
    }
}
