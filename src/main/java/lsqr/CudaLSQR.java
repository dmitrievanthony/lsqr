package lsqr;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

public class CudaLSQR extends AbstractLSQR {

    private final Pointer aPtr = new Pointer();

    private final Pointer bPtr = new Pointer();

    private final int n;

    private final int m;

    private final Pointer uPtr = new Pointer();

    public CudaLSQR(double[] a, double[] b, int n, int m) {
        JCublas.cublasAlloc(n * m, Sizeof.DOUBLE, aPtr);
        JCublas.cublasAlloc(m, Sizeof.DOUBLE, bPtr);
        JCublas.cublasAlloc(m, Sizeof.DOUBLE, uPtr);
        JCublas.cublasSetMatrix(m, n, Sizeof.DOUBLE, Pointer.to(a), m, aPtr, m);
        JCublas.cublasSetVector(m, Sizeof.DOUBLE, Pointer.to(b), 1, bPtr, 1);
        this.n = n;
        this.m = m;
    }

    @Override protected double bnorm() {
        JCublas.cublasDcopy(m, bPtr, 1, uPtr, 1);
        return JCublas.cublasDnrm2(m, bPtr, 1);
    }

    @Override protected double beta(double[] x, double alfa, double beta) {
        Pointer xPtr = new Pointer();
        JCublas.cublasAlloc(n, Sizeof.DOUBLE, xPtr);
        JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(x), 1, xPtr, 1);
        JCublas.cublasDgemv('N', m, n, alfa, aPtr, m, xPtr, 1, beta, uPtr, 1);
        JCublas.cublasFree(xPtr);
        return JCublas.cublasDnrm2(m, uPtr, 1);
    }

    @Override protected double[] iter(double bnorm, double[] target) {
        Pointer tPtr = new Pointer();
        JCublas.cublasAlloc(n, Sizeof.DOUBLE, tPtr);
        JCublas.cublasSetVector(n, Sizeof.DOUBLE, Pointer.to(target), 1, tPtr, 1);
        JCublas.cublasDscal(m, 1 / bnorm, uPtr, 1);
        JCublas.cublasDgemv('T', m, n, 1.0, aPtr, m, uPtr, 1, 1, tPtr, 1);
        JCublas.cublasGetVector(m, Sizeof.DOUBLE, tPtr, 1, Pointer.to(target), 1);
        JCublas.cublasFree(tPtr);
        return target;
    }
}
