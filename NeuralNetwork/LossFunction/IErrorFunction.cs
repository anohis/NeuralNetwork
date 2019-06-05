using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NN.LossFunction
{
    public interface ILossFunction
    {
        double[] Calculate(double[] estimate, double[] real);
        double[] DifferentialExecute(double[] estimate, double[] real);
    }
}
