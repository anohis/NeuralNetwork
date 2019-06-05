using System;

namespace ML.NN.WeightInitilizer
{
    public class WeightInitilizer_Random : IWeightInitilizer
    {
        private Random _random = new Random();
        private int _count; 

        public void Initilize(int count)
        {
            _count = count;
        }

        public double[] GetWeights()
        {
            var list = new double[_count];

            for (int i = 0; i < _count; i++)
            {
                list[i] = _random.NextDouble();
            }

            return list;
        }
    }
}
