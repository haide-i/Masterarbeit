#!/bin/env python3
"""Sends a terminate command to the server.

This script sends a simple terminate command to the BASF2 Geant4 event
server (implemented in server.py), and handles all of the required server
communication that follows.

Authors: Connor Hainje {connor.hainje@pnnl.gov}
         Alex Hagen {alexander.hagen@pnnl.gov}
"""

import zmq
import pandas as pd
from sys import exit


# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://b2ana:237711") # change to match server address

socket.send_json({'cmd':'terminate'})

# Wait for server's params-received status
message = socket.recv()
if message.decode('ascii') != 'RECV0':
    exit('Error: message not delivered successfully.')
    print("Message delivered successfully.")

# Get event data from server
socket.send_string('ready')
binmsg = socket.recv() # <- this data is irrelevant, don't process it
