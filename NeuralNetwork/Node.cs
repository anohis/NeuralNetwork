using System;
using ML.NN.ActivationFunction;

namespace ML.NN
{
    public class Node
    {
        public double Output
        {
            get
            {
                if (_activationFunction == null)
                {
                    return Value;
                }
                else
                {
                    return _activationFunction.Execute(Value);
                }
            }
        }

        public double Value;

        private double[] _weights;
        private double _loss;
        private IActivationFunction _activationFunction = null;

        public Node()
        {
            _weights = new double[0];
        }
        public Node(double[] weights)
        {
            _weights = weights;
        }
        public Node(double[] weights, IActivationFunction activationFunction) : this(weights)
        {
            _activationFunction = activationFunction;
        }

        public void CalculateValue(double[] input)
        {
            if (input.Length != _weights.Length)
            {
                throw new Exception("[Node.CalculateValue] input.Length != Weights.Length");
            }

            Value = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                Value += _weights[i] * input[i];
            }
        }
        public void CalculateLoss(double loss)
        {
            double dOut = 1;
            if (_activationFunction != null)
            {
                dOut = _activationFunction.DifferentialExecute(Value);
            }
            _loss = loss * dOut;
        }
        public double GetLoss(int weightIndex)
        {
            if (_weights.Length <= weightIndex)
            {
                throw new Exception("[Node.GetLoss] _weights.Length <= weightIndex.");
            }
            return _loss * _weights[weightIndex];
        }
        public void Train(double[] preOutput, double learnRate)
        {
            if (preOutput.Length != _weights.Length)
            {
                throw new Exception("[Node.Train] preOutput.Length != _weights.Length.");
            }

            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] -= learnRate * _loss * preOutput[i];
            }
        }
        public override string ToString()
        {
            string str = "weight = {";
            foreach (var v in _weights)
            {
                str += string.Format("{0:0.000},", v);
            }
            str += "}\t";
            str += string.Format("value = {0:0.000}\t output = {1:0.000}\t loss = {2:0.000}", Value, Output, _loss);
            return str;
        }
    }
}
