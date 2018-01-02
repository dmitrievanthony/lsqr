package lsqr;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Random;
import java.util.concurrent.ExecutionException;

public class DataNodeApplication {

    private static final String PORT_PROPERTY_NAME = "port";

    private static final String M_PROPERTY_NAME = "m";

    private static final String N_PROPERTY_NAME = "n";

    public static void main(String... args) throws InterruptedException, ExecutionException, ClassNotFoundException, IOException {
        int port = Integer.valueOf(System.getProperty(PORT_PROPERTY_NAME));
        int m = Integer.valueOf(System.getProperty(M_PROPERTY_NAME));
        int n = Integer.valueOf(System.getProperty(N_PROPERTY_NAME));
        LocalDatasetSegment segment = generateDataSegment(m, n);
        DataNodeServer dataNodeServer = new DataNodeServer(segment, port);
        dataNodeServer.start();
    }

    private static LocalDatasetSegment generateDataSegment(int m, int n) {
        Random random = new Random();
        double[] a = new double[m * n];
        double[] b = new double[m];
        for (int i = 0; i < m; i++) {
            b[i] = random.nextDouble();
            for (int j = 0; j < n; j++) {
                a[j * m + i] = random.nextDouble();
            }
        }
        return new LocalDatasetSegment(a, m, b);
    }

    private static class DataNodeServer {

        private final LocalDatasetSegment segment;

        private final int port;

        public DataNodeServer(LocalDatasetSegment segment, int port) {
            this.segment = segment;
            this.port = port;
        }

        public void start() throws IOException, ExecutionException, InterruptedException, ClassNotFoundException {
            ServerSocket serverSocket = new ServerSocket(port);
            while (true) {
                Socket socket = serverSocket.accept();
                ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
                ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
                int action = ois.readInt();
                switch (action) {
                    case 0: {
                        // calc bnorm
                        double result = segment.bnorm().get();
                        oos.writeDouble(result);
                        oos.flush();
                        break;
                    }
                    case 1: {
                        // calc beta
                        double[] x = (double[]) ois.readObject();
                        double alfa = ois.readDouble();
                        double beta = ois.readDouble();
                        double result = segment.beta(x, alfa, beta).get();
                        oos.writeDouble(result);
                        oos.flush();
                        break;
                    }
                    case 2: {
                        // calc iter
                        double bnorm = ois.readDouble();
                        double[] result = segment.iter(bnorm).get();
                        oos.writeObject(result);
                        oos.flush();
                        break;
                    }
                    default: {
                        throw new RuntimeException("Unknown action " + action);
                    }
                }
                socket.close();
            }
        }
    }
}
