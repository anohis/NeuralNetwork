using System;
using ML.NN.ActivationFunction;
using ML.NN.WeightInitilizer;

namespace ML.NN
{
    public class Layer
    {
        public int NodeCount { get { return _nodes.Length; } }
        public double[] Output
        {
            get
            {
                double[] output = new double[NodeCount];

                for(int i=0;i< _nodes.Length; i++)
                {
                    output[i] = _nodes[i].Output;
                }

                return output;
            }
        }

        private Node[] _nodes;

        public Layer()
        {
            Initialize(0);
        }
        public Layer(int nodeCount)
        {
            Initialize(nodeCount,(index) => 
            {
                _nodes[index] = new Node();
            });
        }
        public Layer(int nodeCount, IWeightInitilizer weightInitilizer)
        {
            Initialize(nodeCount,(index)=> 
            {
                _nodes[index] = new Node(weightInitilizer.GetWeights());
            });
        }
        public Layer(int nodeCount, IWeightInitilizer weightInitilizer, IActivationFunction activationFunction)
        {
            Initialize(nodeCount, (index) =>
            {
                _nodes[index] = new Node(weightInitilizer.GetWeights(), activationFunction);
            });
        }

        public void SetValue(double[] input)
        {
            if (_nodes.Length != input.Length)
            {
                throw new Exception("[Layer.SetValue] _nodes.Length != input.Length");
            }

            for (int i = 0; i < _nodes.Length; i++)
            {
                _nodes[i].Value = input[i];
            }
        }
        public void CalculateValue(double[] input)
        {
            foreach (var node in _nodes)
            {
                node.CalculateValue(input);
            }
        }
        public void CalculateLoss(double[] loss)
        {
            if (_nodes.Length != loss.Length)
            {
                throw new Exception("[Layer.CalculateLoss] _nodes.Length != loss.Length.");
            }

            for (int i = 0; i < loss.Length; i++)
            {
                _nodes[i].CalculateLoss(loss[i]);
            }
        }
        public void CalculateLoss(Layer nextLayer)
        {
            double[] loss = new double[_nodes.Length];

            for (int i = 0; i < _nodes.Length; i++)
            {
                loss[i] = nextLayer.GetLoss(i);
            }

            CalculateLoss(loss);
        }
        public double GetLoss(int weightIndex)
        {
            double loss = 0;

            foreach (var node in _nodes)
            {
                loss += node.GetLoss(weightIndex);
            }

            return loss;
        }
        public void Train(Layer preLayer, double learnRate)
        {
            var preOutput = preLayer.Output;
            foreach (var node in _nodes)
            {
                node.Train(preOutput, learnRate);
            }
        }
        public override string ToString()
        {
            string str = "";
            foreach (var v in _nodes)
            {
                str += string.Format("{0}\n", v.ToString());
            }
            return str;
        }

        private void Initialize(int nodeCount, Action<int> action = null)
        {
            _nodes = new Node[nodeCount];
            for (int i = 0; i < nodeCount; i++)
            {
                action?.Invoke(i);
            }
        }
    }
}
