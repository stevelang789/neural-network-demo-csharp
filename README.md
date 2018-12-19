# Demonstration of a neural network using C#

Taken from Joseph Albahari's LINQPad sample code "Samples for YOW/NDC 2018 - Writing a Neural Net from Scratch", and repackaged as a Visual Studio 2017 solution.

https://www.linqpad.net/RichClient/WritingNeuralNet.zip

## Running it

1. Load the solution in Visual Studio 2017
a. Class library: .NET Framework 4.6.1
b. UI: WPF App, .NET Framework 4.6.1
2. Set `HandwrittenDigitRecognition` as the StartUp Project
3. Press F5!

## Playing around with it

1. Open `Recognizer.cs` under `HandwrittenDigitRecognition`.
2. Change `activator` to `new ReluActivator()` (line 13).
3. Increase `numberOfNeuronsInHiddenLayer` to 30 (line 15).
4. Increase `numberOfEpochs` to 10 (line 17).
5. Add more hidden layers (line 23).
