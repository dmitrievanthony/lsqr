rm data-node-application.pid 2>/dev/null
for port in {4242..4245}
do
	echo Starting data node localhost:$port
	java -server -Xmx8g -Xms8g -cp ./target/lsqr-1.0-SNAPSHOT-jar-with-dependencies.jar -Dport=$port -Dn=500 -Dm=100000 lsqr.DataNodeApplication 2>error.log 1>console.log &
	echo $! >> data-node-application.pid
	if [ -z $data ]; then data=localhost:$port; else data=$data,localhost:$port; fi
done
echo Waiting data nodes to be initialized
sleep 5
echo Starting client node
java -server -Xmx8g -Xms8g -cp ./target/lsqr-1.0-SNAPSHOT-jar-with-dependencies.jar -Dn=500 -Ddata=$data -Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.NativeRefBLAS lsqr.Application
cat data-node-application.pid | xargs kill -9
