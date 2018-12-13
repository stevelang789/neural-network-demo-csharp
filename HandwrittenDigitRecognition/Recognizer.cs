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
            const int neuronsInInputLayer = Visualizer.ImageWidthHeight * Visualizer.ImageWidthHeight;
            const int neuronsInOutputLayer = 10;

            var network = new Network(
                new ReluActivator(),
                neuronsInInputLayer,
                10,
                neuronsInOutputLayer);

            var trainer = new Trainer(network);
            var panel = Visualizer.NeuralNetRenderer(trainer, centreImage: true);

            attachToUi(panel);

            var trainingData = ImageSample.LoadTrainingImages();   // 50,000 training images
            var testingData = ImageSample.LoadTestingImages();     // 10,000 testing images

            await Task.Run(() => trainer.Train(trainingData, testingData, learningRate: .01, epochs: 10));
        }
    }
}
