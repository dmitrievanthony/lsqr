package lsqr;

import com.github.fommil.netlib.BLAS;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

public class Dataset {

    private static BLAS blas = BLAS.getInstance();

    private final DatasetSegment[] segments;

    public Dataset(DatasetSegment[] segments) {
        this.segments = segments;
    }

    @SuppressWarnings("unchecked")
    public double bnorm() throws ExecutionException, InterruptedException {
        double[] bnorm = new double[segments.length];
        Future<Double>[] bnormFeatures = new Future[segments.length];
        for (int i = 0; i < segments.length; i++)
            bnormFeatures[i] = segments[i].bnorm();
        for (int i = 0; i < segments.length; i++)
            bnorm[i] = bnormFeatures[i].get();
        return blas.dnrm2(bnorm.length, bnorm, 1);
    }

    @SuppressWarnings("unchecked")
    public double beta(double[] x0, double alfa, double b) throws ExecutionException, InterruptedException {
        double[] beta = new double[segments.length];
        Future<Double>[] betaFeatures = new Future[segments.length];
        for (int i = 0; i < segments.length; i++)
            betaFeatures[i] = segments[i].beta(x0, alfa, b);
        for (int i = 0; i < segments.length; i++)
            beta[i] = betaFeatures[i].get();
        return blas.dnrm2(beta.length, beta, 1);
    }

    @SuppressWarnings("unchecked")
    public double[] iter(double bnorm, double[] target) throws ExecutionException, InterruptedException {
        Future<double[]>[] features = new Future[segments.length];
        for (int i = 0; i < segments.length; i++)
            features[i] = segments[i].iter(bnorm);
        for (int i = 0; i < segments.length; i++) {
            double[] vi = features[i].get();
            if (vi != null) {
                blas.daxpy(target.length, 1.0, vi, 1, target, 1);
            }
        }
        return target;
    }
}
