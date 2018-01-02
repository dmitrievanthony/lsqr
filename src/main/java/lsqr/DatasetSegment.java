package lsqr;

import java.util.concurrent.Future;

public interface DatasetSegment {

    Future<Double> bnorm();

    Future<Double> beta(double[] x, double alfa, double beta);

    Future<double[]> iter(double bnorm);
}
