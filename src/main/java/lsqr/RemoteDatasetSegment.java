package lsqr;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class RemoteDatasetSegment implements DatasetSegment {

    private static ExecutorService service = Executors.newFixedThreadPool(10);

    private String host;

    private int port;

    public RemoteDatasetSegment(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public Future<Double> bnorm() {
        return service.submit(() -> {
            Socket socket = new Socket(host, port);
            ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
            ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
            oos.writeInt(0);
            oos.flush();
            double result = ois.readDouble();
            socket.close();
            return result;
        });
    }

    @Override
    public Future<Double> beta(double[] x, double alfa, double beta) {
        return service.submit(() -> {
            Socket socket = new Socket(host, port);
            ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
            ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
            oos.writeInt(1);
            oos.writeObject(x);
            oos.writeDouble(alfa);
            oos.writeDouble(beta);
            oos.flush();
            double result = ois.readDouble();
            socket.close();
            return result;
        });
    }

    @Override
    public Future<double[]> iter(double bnorm) {
        return service.submit(() -> {
            Socket socket = new Socket(host, port);
            ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
            ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
            oos.writeInt(2);
            oos.writeDouble(bnorm);
            oos.flush();
            double[] result = (double[]) ois.readObject();
            socket.close();
            return result;
        });
    }
}
