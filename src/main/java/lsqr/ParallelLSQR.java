package lsqr;

import com.github.fommil.netlib.BLAS;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Parallel implementation of LSQR algorithm with assumption than data is stored locally and processed in given number
 * of threads.
 */
public class ParallelLSQR extends AbstractLSQR {
    /** */
    private static final BLAS blas = BLAS.getInstance();

    private final PartitionContext[] partitionContexts;

    /** */
    public ParallelLSQR(double[][] a, double[] b, int nThreads) {
        partitionContexts = new PartitionContext[nThreads];
        int partSize = a.length / nThreads;
        ExecutorService executorSvc = Executors.newFixedThreadPool(nThreads);
        for (int i = 0; i < nThreads; i++) {
            int actualPartSize = (i + 1) * partSize <= a.length ? partSize : a.length - i * partSize;
            double[][] partA = new double[actualPartSize][];
            double[] partB = new double[actualPartSize];
            for (int j = 0; j < actualPartSize; j++) {
                partA[j] = a[i * partSize + j];
                partB[j] = b[i * partSize + j];
            }
            partitionContexts[i] = new PartitionContext(executorSvc, partA, partB, a[0].length, actualPartSize);
        }
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override protected double bnorm() {
        Future<Double>[] futures = new Future[partitionContexts.length];
        for (int i = 0; i < partitionContexts.length; i++)
            futures[i] = partitionContexts[i].bnorm();
        double[] res = new double[futures.length];
        for (int i = 0; i < futures.length; i++) {
            try {
                res[i] = futures[i].get();
            }
            catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
        return blas.dnrm2(res.length, res, 1);
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override protected double beta(double[] x, double alfa, double beta) {
        Future<Double>[] futures = new Future[partitionContexts.length];
        for (int i = 0; i < partitionContexts.length; i++)
            futures[i] = partitionContexts[i].beta(x, alfa, beta);
        double[] res = new double[futures.length];
        for (int i = 0; i < futures.length; i++) {
            try {
                res[i] = futures[i].get();
            }
            catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
        return blas.dnrm2(res.length, res, 1);
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override protected double[] iter(double bnorm, double[] target) {
        Future<double[]>[] futures = new Future[partitionContexts.length];
        for (int i = 0; i < partitionContexts.length; i++)
            futures[i] = partitionContexts[i].iter(bnorm);
        for (int i = 0; i < futures.length; i++) {
            try {
                blas.daxpy(target.length, 1.0, futures[i].get(), 1, target, 1);
            }
            catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }

        return target;
    }

    private static class PartitionContext {

        private final ExecutorService executorSvc;

        private final double[] a;

        private final double[] b;

        private final int n;

        private final int m;

        private double[] u;

        public PartitionContext(ExecutorService executorSvc, double[][] a, double[] b, int n, int m) {
            this.executorSvc = executorSvc;
            this.a = flat(a);
            this.b = b;
            this.n = n;
            this.m = m;
        }

        /** */
        protected Future<Double> bnorm() {
            return executorSvc.submit(() -> {
                u = Arrays.copyOf(b, b.length);
                return blas.dnrm2(b.length, b, 1);
            });
        }

        /** */
        protected Future<Double> beta(double[] x, double alfa, double beta) {
            return executorSvc.submit(() -> {
                blas.dgemv("N", m, n, alfa, a, m, x, 1, beta, u, 1);
                return blas.dnrm2(u.length, u, 1);
            });
        }

        /** */
        protected Future<double[]> iter(double bnorm) {
            return executorSvc.submit(() -> {
                blas.dscal(u.length, 1 / bnorm, u, 1);
                // target = A^t* + target
                double[] v = new double[n];
                blas.dgemv("T", m, n, 1.0, a, m, u, 1, 0, v, 1);
                return v;
            });
        }

        /** */
        private static double[] flat(double[][] x) {
            double[] res = new double[x.length * x[0].length];
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++)
                    res[j * x.length + i] = x[i][j];
            }
            return res;
        }

        @Override public String toString() {
            return "PartitionContext{" +
                "a=" + Arrays.toString(a) +
                ", b=" + Arrays.toString(b) +
                ", n=" + n +
                ", m=" + m +
                '}';
        }
    }
}