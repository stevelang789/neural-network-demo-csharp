using System;
using System.Threading.Tasks;
using System.Windows;
using NeuralNetwork;
using NeuralNetwork.Activators;

namespace HandwrittenDigitRecognition
{
    internal class Recognizer
    {
        public async Task Run(Func<UIElement, int> attachToUi)
        {
            var activator = new LogisticSigmoidActivator(); // ReluActivator seems to be the best one
            const int numberOfNeuronsInInputLayer = Visualizer.ImageWidthHeight * Visualizer.ImageWidthHeight;
            const int numberOfNeuronsInHiddenLayer = 10; // Increase this to 30 for best results
            const int numberOfNeuronsInOutputLayer = 10;
            const int numberOfEpochs = 5; // More than 5 doesn't seem to make any difference

            var network = new Network(
                activator,
                numberOfNeuronsInInputLayer,
                numberOfNeuronsInHiddenLayer,
                // Optionally add more hidden layers by specifying the number of neurons in each layer: 10, 10,
                numberOfNeuronsInOutputLayer);

            var trainer = new Trainer(network);
            var panel = Visualizer.NeuralNetRenderer(trainer, centreImage: true);

            attachToUi(panel);

            var trainingData = ImageSample.LoadTrainingImages();   // 50,000 training images
            var testingData = ImageSample.LoadTestingImages();     // 10,000 testing images

            await Task.Run(() => trainer.Train(
                trainingData,
                testingData,
                learningRate: .01,
                epochs: numberOfEpochs));
        }
    }
}
