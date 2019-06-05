using System;
using System.Collections.Generic;
using System.Linq;
using ML.NN.ActivationFunction;
using ML.NN.LossFunction;
using ML.NN.WeightInitilizer;

namespace ML.NN
{
    public class NeuralNetwork
    {
        private double _learnRate;
        private IActivationFunction _activationFunction;
        private IWeightInitilizer _weightInitilizer;
        private ILossFunction _lossFunction;

        private List<Layer> Layers;

        public NeuralNetwork(double learnRate,IActivationFunction activationFunction, 
            IWeightInitilizer weightInitilizer,ILossFunction lossFunction)
        {
            if (activationFunction == null)
            {
                throw new Exception("[NeuralNetwork] activationFunction is null.");
            }
            if (weightInitilizer == null)
            {
                throw new Exception("[NeuralNetwork] weightInitilizer is null.");
            }
            if (lossFunction == null)
            {
                throw new Exception("[NeuralNetwork] lossFunction is null.");
            }

            _learnRate = learnRate;
            _activationFunction = activationFunction;
            _weightInitilizer = weightInitilizer;
            _lossFunction = lossFunction;

            Layers = new List<Layer>();
        }

        public void AddLayer(int nodeCount)
        {
            Layer layer = null;

            if (Layers.Count <= 0)
            {
                layer = new Layer(nodeCount);
            }
            else
            {
                _weightInitilizer.Initilize(Layers.Last().NodeCount);
                layer = new Layer(nodeCount, _weightInitilizer, _activationFunction);
            }

            Layers.Add(layer);
        }
        public double[] Calculate(double[] input)
        {
            if (Layers.Count <= 0)
            {
                throw new Exception("[NeuralNetwork.Calculate] Layers.Count <= 0.");
            }

            Layers.First().SetValue(input);

            var output = Layers.First().Output;

            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                layer.CalculateValue(output);
                output = layer.Output;
            }

            return output;
        }
        public void Train(double[] input, double[] real)
        {
            var output = Calculate(input);
            if (output.Length != real.Length)
            {
                throw new Exception("[NeuralNetwork.Train] output.Length != real.Length.");
            }

            var loss =  _lossFunction.DifferentialExecute(output, real);

            Layers.Last().CalculateLoss(loss);

            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                Layers[i].CalculateLoss(Layers[i + 1]);
            }

            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].Train(Layers[i - 1], _learnRate);
            }
        }
        public override string ToString()
        {
            string str = "";
            for (int i = 0; i < Layers.Count; i++)
            {
                str += string.Format("====================\nLayer : {0}\n{1}",i, Layers[i].ToString());
            }
            return str;
        }
    }
}
