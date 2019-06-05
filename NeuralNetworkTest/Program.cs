using System;
using ML.NN;
using ML.NN.ActivationFunction;
using ML.NN.LossFunction;
using ML.NN.WeightInitilizer;

namespace NeuralNetworkTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var activationFunction = new ActivationFunction_Sigmoid();
            var weightInitilizer = new WeightInitilizer_Random();
            var lossFunction = new LossFunction_Variance();
            var learnRate = 0.5f;

            var nn = new NeuralNetwork(learnRate, activationFunction, weightInitilizer, lossFunction);
            nn.AddLayer(2);
            nn.AddLayer(2);

            while(true)
            {
                var input = new double[] { 0.1, 0.9 };
                var real = new double[] { 0.9, 0.1 };
                nn.Train(input, real);
                Console.Write(nn.ToString());
                Console.ReadLine();
            }
        }
    }
}
