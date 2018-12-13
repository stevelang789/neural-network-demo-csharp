﻿using System;

namespace NeuralNetwork
{
    public class Sample
    {
        public double[] Data;
        public double[] ExpectedOutput;
        public Func<double[], bool> IsOutputCorrect;
    }
}
