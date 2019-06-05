namespace ML.NN.ActivationFunction
{
    public interface IActivationFunction
    {
        double Execute(double value);
        double DifferentialExecute(double value);
    }
}
