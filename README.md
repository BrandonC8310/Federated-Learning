# COMP3221 ASM2
Before running the program, please firstly run the following command (for MacOS) to change the maximum udp transporting data size:

`sudo sysctl -w net.inet.udp.maxdgram=65535`

## Run the server firstly.
There are two options to run the server:
### Option 1:
  `python3 COMP3221_FLServer.py 6000 0` (use port number 6000 for the server and aggregates all the client models)

### Option 2:
`python3 COMP3221_FLServer.py 6000 1` (use port number 6000 for the server and aggregates random 2 client models from the five models)

## Run the five clients Secondly.

There are two options to run a client:
### Option 1:
`python3 COMP3221_FLClient.py client1 6001 0` (use port number 6001 for the client with id "client1" , and use GD)

### Option2:
`python3 COMP3221_FLClient.py client1 6001 1` (use port number 6001 for the client with id "client1" , and use Mini-Batch GD)

### An example to run all the five clients:
`python3 COMP3221_FLClient.py client1 6001 0` 

`python3 COMP3221_FLClient.py client2 6002 0`

`python3 COMP3221_FLClient.py client3 6003 0`

`python3 COMP3221_FLClient.py client4 6004 1`

`python3 COMP3221_FLClient.py client5 6005 1`

This indicates that we use GD for client1, client2, and client3. And we use Mini-Batch GD for client 4 and client 5.

## After the server and the clients finished executing, run the evaluation program:
`python3 COMP3221_FLEvaluation.py`

This will output diagrams of the overall performance of the global model (loss and accuracy). 

## What we've done
Everything including addressing the client failiure issue.
