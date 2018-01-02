package lsqr;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutionException;

public class Application {

    private static final String DATA_NODES_PROPERTY_NAME = "data";

    private static final String N_PROPERTY_NAME = "n";

    public static void main(String... args) throws ExecutionException, InterruptedException, FileNotFoundException {
//        File file = new File("/Users/antondmitriev/Downloads/data/ethylene_methane.txt");
//        List<double[]> dataset = new ArrayList<>();
//        Scanner scanner = new Scanner(file);
//        scanner.nextLine();
//        while (scanner.hasNextLine()) {
//            String[] line = scanner.nextLine().split("\\s+");
//            double[] fields = new double[line.length];
//            for (int i = 0; i < line.length; i++) {
//                fields[i] = Double.valueOf(line[i]);
//            }
//            dataset.add(fields);
//        }
//        int m = dataset.size();
//        int n = 16;
//        double[] a = new double[m * (n + 1)];
//        double[] b = new double[m];
//        for (int i = 0; i < dataset.size(); i++) {
//            double[] fields = dataset.get(i);
//            b[i] = fields[1];
//            a[i] = 1.0;
//            for (int j = 0; j < n; j++) {
//                a[(j + 1) * m + i] = fields[3 + j];
//            }
//        }
//        System.out.println("Calculating");
//        long t1 = System.currentTimeMillis();
//        for (int i = 0; i < 10; i++) {
//            LSQR.lsqr(a, m, b, 0, 1e-6, 1e-6, 1e8, -1, false, null);
//        }
//        long t2 = System.currentTimeMillis();
//        System.out.println("Time = " + (t2 - t1) / 10.0 / 1000.0 + "s");

        String[] dataNodes = System.getProperty(DATA_NODES_PROPERTY_NAME).split(",");
        RemoteDatasetSegment[] segments = new RemoteDatasetSegment[dataNodes.length];
        int n = Integer.valueOf(System.getProperty(N_PROPERTY_NAME));
        for (int i = 0; i < segments.length; i++) {
            String[] hostAndPort = dataNodes[i].split(":");
            segments[i] = new RemoteDatasetSegment(hostAndPort[0], Integer.valueOf(hostAndPort[1]));
        }
        Dataset dataset = new Dataset(segments);
        long t1 = System.currentTimeMillis();
        double[] res2 = null;
        for (int i = 0; i < 10; i++) {
            res2 = DLSQR.lsqr(dataset, n, 0, 1e-6, 1e-6, 1e8, -1, false, null);
        }
        long t2 = System.currentTimeMillis();
        System.out.println("Time = " + (t2 - t1) / 10.0 + "ms");



        int m = 400_000;
        int ln = 500;

        double[][] a = new double[m][n];
        double[] b = new double[m];
        Random random = new Random();
        for (int i = 0; i < m; i++) {
            b[i] = random.nextDouble();
            for (int j = 0; j < n; j++) {
                a[i][j] = random.nextDouble();
            }
        }

        double[] aa = flat(a);
        t1 = System.currentTimeMillis();
        for (int i = 0; i < 10; i++) {
            double[] res1 = LSQR.lsqr(aa, m, b, 0, 1e-6, 1e-6, 1e8, -1, false, null);
        }
        t2 = System.currentTimeMillis();
        System.out.println("Local " + (t2 - t1) / 10.0 + "ms");

        dataset = makeDataset(a, b, 4);
        long t3 = System.currentTimeMillis();
        for (int i = 0; i < 10; i++) {
            res2 = DLSQR.lsqr(dataset, n, 0, 1e-6, 1e-6, 1e8, -1, false, null);
        }
        long t4 = System.currentTimeMillis();
        System.out.println("Parallel " + (t4 - t3) / 10.0 + "ms");
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

    private static Dataset makeDataset(double[][] x, double[] y, int count) {
        LocalDatasetSegment[] segments = new LocalDatasetSegment[count];
        for (int i = 0; i < count; i++) {
            if (i == count - 1) {
                double[][] xSegment = Arrays.copyOfRange(x, i * (x.length/count), x.length);
                double[] ySegment = Arrays.copyOfRange(y, i * (x.length/count), x.length);
                int m = x.length - i * (x.length/count);
                segments[i] = new LocalDatasetSegment(flat(xSegment), m, ySegment);
            }
            else {
                double[][] xSegment = Arrays.copyOfRange(x, i * (x.length/count), (i+1)*(x.length/count));
                double[] ySegment = Arrays.copyOfRange(y, i * (x.length/count), (i+1)*(x.length/count));
                int m = (i+1)*(x.length/count) - i * (x.length/count);
                segments[i] = new LocalDatasetSegment(flat(xSegment), m, ySegment);
            }
        }
        return new Dataset(segments);
    }
}
