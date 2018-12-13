using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Trainer
    {
        public readonly Network Network;
        public int CurrentEpoch;
        public double CurrentAccuracy;
        public int Iterations;
        public string TrainingInfo;
        public string Message;

        private readonly CancellationTokenSource _cancelSource = new CancellationTokenSource();
        private ParallelOptions CancellableParallel => new ParallelOptions { CancellationToken = _cancelSource.Token };

        public Trainer(Network network) => Network = network;

        public void Train(Sample[] trainingData, Sample[] testingData, double learningRate, int epochs)
        {
            var trainingSet = trainingData.ToArray();

            TrainingInfo = $"Learning rate = {learningRate}";

            for (CurrentEpoch = 0; CurrentEpoch < epochs; CurrentEpoch++)
            {
                CurrentAccuracy = TrainEpoch(trainingSet, learningRate);
                learningRate *= .9;   // This help to avoids oscillation as our accuracy improves.
            }

            var testAccuracy = ((Test(new FiringNetwork(Network), testingData) * 100).ToString("N1") + "%");
            TrainingInfo += $"\r\nTotal epochs = {CurrentEpoch}\r\nFinal test accuracy = {testAccuracy}";

            Message = "Done!";
        }

        private double TrainEpoch(Sample[] trainingData, double learningRate)
        {
            Shuffle(new Random(), trainingData);   // For each training epoch, randomize order of the training samples.

            // One FiringNetwork per thread to avoid thread-safety problems.
            var trainer = new ThreadLocal<FiringNetwork>(() => new FiringNetwork(Network));
            Parallel.ForEach(trainingData, CancellableParallel, sample =>
            {
                trainer.Value.Learn(sample.Data, sample.ExpectedOutput, learningRate);
                Interlocked.Increment(ref Iterations);
            });

            return Test(new FiringNetwork(Network), trainingData.Take(10000).ToArray()) * 100;
        }

        private static double Test(FiringNetwork firingNetwork, Sample[] samples)
        {
            int bad = 0, good = 0;
            foreach (var sample in samples)
            {
                firingNetwork.FeedForward(sample.Data);
                if (sample.IsOutputCorrect(firingNetwork.OutputValues.ToArray()))
                    good++;
                else
                    bad++;
            }
            return (double)good / (good + bad);
        }

        private static void Shuffle<T>(Random random, T[] array)
        {
            var n = array.Length;
            while (n > 1)
            {
                var k = random.Next(n--);
                var temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
