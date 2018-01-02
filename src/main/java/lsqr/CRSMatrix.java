package lsqr;

public class CRSMatrix {

    private final double[] val;

    private final int[] colInd;

    private final int[] rowPtr;

    private final int m, n;

    public CRSMatrix(double[][] denseMatrix) {
        m = denseMatrix.length;
        n = denseMatrix[0].length;
        int nonEmptyElements = 0;
        for (int i = 0; i < denseMatrix.length; i++)
            for (int j = 0; j < denseMatrix[i].length; j++)
                if (denseMatrix[i][j] != 0)
                    nonEmptyElements++;
        val = new double[nonEmptyElements];
        colInd = new int[nonEmptyElements];
        rowPtr = new int[denseMatrix.length];
        int ptr = 0;
        for (int i = 0; i < denseMatrix.length; i++) {
            rowPtr[i] = ptr;
            for (int j = 0; j < denseMatrix[i].length; j++)
                if (denseMatrix[i][j] != 0) {
                    val[ptr] = denseMatrix[i][j];
                    colInd[ptr] = j;
                    ptr++;
                }
        }
    }

    public double[] matvec(double[] vector) {
        double[] res = new double[m];
        for (int i = 0; i < rowPtr.length; i++) {
            int from = rowPtr[i];
            int to = i == rowPtr.length - 1 ? val.length : rowPtr[i + 1];
            for (int j = from; j < to; j++) {
                res[i] += val[j] * vector[colInd[j]];
            }
        }
        return res;
    }

    public double[] rmatvec(double[] vector) {
        double[] res = new double[n];
        for (int i = 0; i < rowPtr.length; i++) {
            int from = rowPtr[i];
            int to = i == rowPtr.length - 1 ? val.length : rowPtr[i + 1];
            for (int j = from; j < to; j++) {
                res[colInd[j]] += val[j] * vector[i];
            }
        }
        return res;
    }
}
