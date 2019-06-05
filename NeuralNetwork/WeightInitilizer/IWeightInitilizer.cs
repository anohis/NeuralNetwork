namespace ML.NN.WeightInitilizer
{
    public interface IWeightInitilizer
    {
        void Initilize(int count);
        double[] GetWeights();
    }
}
