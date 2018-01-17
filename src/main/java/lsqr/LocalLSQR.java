package lsqr;

import com.github.fommil.netlib.BLAS;
import java.util.Arrays;

/**
 * Local implementation of LSQR algorithm with assumption than data is stored locally and processed in one thread.
 */
public class LocalLSQR extends AbstractLSQR {
    /** */
    private static final BLAS blas = BLAS.getInstance();

    /** */
    private final double[] a;

    /** */
    private final double[] b;

    /** */
    private final int n;

    /** */
    private final int m;

    /** */
    private double[] u;

    /** */
    public LocalLSQR(double[][] a, double[] b) {
        this.a = flat(a);
        this.b = b;
        this.n = a[0].length;
        this.m = a.length;
    }

    /** {@inheritDoc} */
    @Override protected double bnorm() {
        u = Arrays.copyOf(b, b.length);
        return blas.dnrm2(b.length, b, 1);
    }

    /** {@inheritDoc} */
    @Override protected double beta(double[] x, double alfa, double beta) {
        blas.dgemv("N", m, n, alfa, a, m, x, 1, beta, u, 1);
        return blas.dnrm2(u.length, u, 1);
    }

    /** {@inheritDoc} */
    @Override protected double[] iter(double bnorm, double[] target) {
        blas.dscal(u.length, 1 / bnorm, u, 1);
        // target = A^t* + target
        blas.dgemv("T", m, n, 1.0, a, m, u, 1, 1, target, 1);
        return target;
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
}