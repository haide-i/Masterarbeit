#!/bin/env python3
"""Sends parameters to server and receives resulting data.

This script manages the sending of module parameters to the BASF2 Geant4
event server (implemented in server.py). It then handles the receipt and 
decoding of the resulting data. Users should edit the definition of the 
`params` variable, and add whatever code they desire at the end to use 
the returned DataFrame.

Authors: Connor Hainje {connor.hainje@pnnl.gov}
         Alex Hagen (alexander.hagen@pnnl.gov}
"""

import zmq
import pandas as pd
from sys import exit


def send_params(params):
    """Sends parameters to server.

    Sends the desired parameter dict `params` to the server as JSON, and
    handles the server's received status. Exits if server status is not 0.

    Args:
        params: A dict mapping module names to dicts of parameter name,
          value pairs. 
    """
    socket.send_json(params)
    message = socket.recv()
    if message.decode('ascii') != 'RECV0':
        exit('Error: message not delivered successfully.')
        print('Message delivered successfully.')


def get_df(key="data"):
    """Gets event data back from the server.

    Sends the server a request for data, then receives and processes the
    returned binary data. Returns a Pandas DataFrame.

    Args:
        key (str): The key used in sim_server to make the DataFrame store.
          Defaults to "data", the default value in server.py.
    """
    socket.send_string('ready')
    binmsg = socket.recv()
    store = pd.HDFStore(
        'tmp.h5', 
        mode="r",
        driver="H5FD_CORE",
        driver_core_backing_store=0,
        driver_core_image=binmsg
    )
    df = store[key] 
    store.close()
    return df


# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://b2ana:237711") # change to match server address

# Specify parameters for each module
params = {
    'ParticleGun': {
        'pdgCodes': [-211],
        'nTracks': 1,
        'varyNTracks': False,
        'independentVertices': False,
        'momentumGeneration': 'fixed',
        'momentumParams': [2.0],
        'thetaGeneration': 'fixed',
        'thetaParams': [87.],
        'phiGeneration': 'fixed',
        'phiParams': [2.5],
        'vertexGeneration': 'fixed',
        'xVertexParams': 0.,
        'yVertexParams': 0.,
        'zVertexParams': 0.
    }
}

send_params(params)
df = get_df()

# Do what you want with the recovered DataFrame here
for p in df.head().iterrows():
    print(p)

