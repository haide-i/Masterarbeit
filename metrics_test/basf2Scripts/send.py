import zmq
import json
import pandas as pd
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://b2ana:237711")

# data = dict(angularDistribution='cone', numPhotons=10)
data = dict(cmd='terminate')
socket.send_json(data)
message = socket.recv()
if message.decode('ascii') == 'RECV0':
        print("message delivered successfully")
        socket.send_string('ready')
        binmsg = socket.recv()
        key = 'photons'
        store = pd.HDFStore('tmp.h5', mode="r",
                            driver="H5FD_CORE",
                                                        driver_core_backing_store=0,
                                                                            driver_core_image=binmsg)
        df = store[key]
        for p in df.head().iterrows():
            print(p)
