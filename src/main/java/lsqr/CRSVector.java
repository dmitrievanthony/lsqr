package lsqr;

public class CRSVector {

    private final double[] val;

    private final int[] colInd;

    private final int n;

    public CRSVector(double[] denseVector) {
        n = denseVector.length;
        int nonEmptyElements = 0;
        for (int i = 0; i < denseVector.length; i++)
            if (denseVector[i] != 0)
                nonEmptyElements++;
        val = new double[nonEmptyElements];
        colInd = new int[nonEmptyElements];
        int ptr = 0;
        for (int i = 0; i < denseVector.length; i++)
            if (denseVector[i] != 0) {
                val[ptr] = denseVector[i];
                colInd[ptr] = i;
                ptr++;
            }
    }

    public double[] getVal() {
        return val;
    }

    public int[] getColInd() {
        return colInd;
    }

    public int getN() {
        return n;
    }
}
