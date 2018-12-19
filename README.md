# Demonstration of a neural network using C#

Taken from Joseph Albahari's LINQPad sample code "Samples for YOW/NDC 2018 - Writing a Neural Net from Scratch", and repackaged as a Visual Studio 2017 solution.

https://www.linqpad.net/RichClient/WritingNeuralNet.zip

## Running it

1. Load the solution in Visual Studio 2017
   - Class library: .NET Framework 4.6.1
   - UI: WPF App, .NET Framework 4.6.1
2. Set `HandwrittenDigitRecognition` as the StartUp Project
3. Press F5!

## Playing around with it

Open `Recognizer.cs` in the `HandwrittenDigitRecognition` project.

- Change `activator` to `new ReluActivator()` (line 13).
- Increase `numberOfNeuronsInHiddenLayer` to 30 (line 15).
- Increase `numberOfEpochs` to 10 (line 17).
- Add more hidden layers (line 23).
