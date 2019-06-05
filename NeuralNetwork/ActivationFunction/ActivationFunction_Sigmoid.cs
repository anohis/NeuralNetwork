using System;

namespace ML.NN.ActivationFunction
{
    public class ActivationFunction_Sigmoid : IActivationFunction
    {
        public double DifferentialExecute(double value)
        {
            value = Execute(value);
            return value * (1 - value);
        }

        public double Execute(double value)
        {
            return 1.0f / (1 + Math.Pow(Math.E, -value));
        }
    }

}
