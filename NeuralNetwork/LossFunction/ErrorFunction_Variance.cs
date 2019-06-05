using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NN.LossFunction
{
    public class LossFunction_Variance : ILossFunction
    {
        public double[] Calculate(double[] estimate, double[] real)
        {
            if (estimate.Length != real.Length)
            {
                throw new Exception("[LossFunction_Variance.Calculate] estimate.Length != real.Length.");
            }

            double[] error = new double[estimate.Length];
            for (int i = 0; i < estimate.Length; i++)
            {
                error[i] = Math.Pow(real[i] - estimate[i], 2) / 2;
            }

            return error;
        }

        public double[] DifferentialExecute(double[] estimate, double[] real)
        {
            if (estimate.Length != real.Length)
            {
                throw new Exception("[LossFunction_Variance.DifferentialExecute] estimate.Length != real.Length.");
            }

            double[] error = new double[estimate.Length];
            for (int i = 0; i < estimate.Length; i++)
            {
                error[i] = -(real[i] - estimate[i]);
            }

            return error;
        }
    }
}
