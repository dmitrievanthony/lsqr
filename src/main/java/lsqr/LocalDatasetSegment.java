package lsqr;

import com.github.fommil.netlib.BLAS;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class LocalDatasetSegment implements DatasetSegment {

    private static BLAS blas = BLAS.getInstance();

    private static ExecutorService service = Executors.newFixedThreadPool(4);

    private final double[] a;

    private final int m;

    private final int n;

    private final double[] b;

    private volatile double[] u;

    public LocalDatasetSegment(double[] a, int m, double[] b) {
        this.a = a;
        this.m = m;
        this.n = a.length / m;
        this.b = b;
    }

    @Override
    public Future<Double> bnorm() {
        return service.submit(() -> {
            u = Arrays.copyOf(b, b.length);
            return blas.dnrm2(b.length, b, 1);
        });
    }

    @Override
    public Future<Double> beta(double[] x, double alfa, double beta) {
        return service.submit(() -> {
            blas.dgemv("N", m, n, alfa, a, m, x, 1, beta, u, 1);
            return blas.dnrm2(u.length, u, 1);
        });
    }

    @Override
    public Future<double[]> iter(double bnorm) {
        return service.submit(() -> {
            blas.dscal(u.length, 1 / bnorm, u, 1);
            double[] v = new double[n];
            blas.dgemv("T", m, n, 1.0, a, m, u, 1, 0, v, 1);
            return v;
        });
    }
}
