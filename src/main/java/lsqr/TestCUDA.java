package lsqr;

import java.util.Random;

public class TestCUDA {

    public static void main(String... args) {
        int m = Integer.parseInt(System.getProperty("m"));
        int n = Integer.parseInt(System.getProperty("n"));
        double[][] a = new double[m][n];
        double[] b = new double[m];

        Random random = new Random();
        for (int i = 0; i < m; i++) {
            b[i] = random.nextDouble();
            for (int j = 0; j < n; j++) {
                a[i][j] = random.nextDouble();
            }
        }

        double[] aFlat = flat(a);
        long t1 = System.currentTimeMillis();
        for (int i = 0; i < 20; i++) {
            LSQR.lsqr(aFlat, m, b, 0, 1e-6, 1e-6, 1e8, -1, false, null);
        }
        long t2 = System.currentTimeMillis();
        System.out.println("Local CPU time : " + (t2 - t1) / 10.0 + "ms");

        CudaLSQR lsqr = new CudaLSQR(aFlat, b, n, m);
        t1 = System.currentTimeMillis();
        for (int i = 0; i < 20; i++) {
            lsqr.solve(n, 0, 1e-6, 1e-6, 1e8, -1, false, null);
        }
        t2 = System.currentTimeMillis();
        System.out.println("Local GPU time : " + (t2 - t1) / 10.0 + "ms");
    }

    private static double[] flat(double[][] x) {
        double[] result = new double[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[j * x.length + i] = x[i][j];
            }
        }
        return result;
    }
}
