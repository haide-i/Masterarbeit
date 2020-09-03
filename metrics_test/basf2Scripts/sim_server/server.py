#!/bin/env python3
"""Creates a BASF2 Geant4 server for repeated single-event simulation.

This script creates and runs a simple ZMQ reply server that can modify the 
parameters of specific, pre-determined moudles on an event-by-event basis,
eliminating the need to restart the simulation framework and granting
greater flexibility.

Authors: Connor Hainje {connor.hainje@pnnl.gov}
         Alex Hagen {alexander.hagen@pnnl.gov}
"""

import basf2 as b2
from ROOT import Belle2

import zmq
import pandas as pd
import logging
import uuid


class Server(b2.Module):
    """Module that maintains a ZMQ server and manages module parameters.

    Attributes:
        context: The ZMQ Context object for establishing the socket.
        socket: Our ZMQ reply socket for communication with clients.
        eis: A reference to the EventInfoSetter module, so that we can 
          shut down the server.
        shut_down (bool): Tracks whether a terminate signal has been 
          received from the client.
        module_params: A dict mapping module names to dicts of parameter
          name, value pairs.
    """

    def __init__(self, eventinfosetter, port=237711):
        """Inits the Server module and sets up the server.
        
        Args:
            eventinfosetter: A reference to the EventInfoSetter module.
            port (int): The port to which the server socket is bound. 
              Defaults to 237711.
        """
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.eis = eventinfosetter
        self.shut_down = False
        self.module_params = dict()

    def event(self):
        """Receives new module parameters from client.

        Receives a new set of module parameters from the client and sends
        a success response on receipt. If the parameters contain a 
        terminate command, it sets the EventInfoSetter `evtNumList` to 1, 
        so the server will shut down after this event's completion.
        Otherwise, the module_params attribute is updated. Note that this
        method will block indefinitely until a new message is sent by 
        the client.
        """
        data = self.socket.recv_json()
        self.socket.send_string("RECV0")
        
        if ('cmd' in data) and (data['cmd'] == 'terminate'):
            logging.debug('Turning off the EventInfoSetter')
            self.eis.param('evtNumList', [1])
            self.shut_down = True
            return

        for m_name, m_params in data.items():
            self.module_params[m_name] = m_params

    def get_params(self, module_name):
        """Gets a module's current parameter dict.

        Gets a module's parameter dict from the attribute module_params.
        If no such dict exists (as would be the case if the client has 
        not sent any parameters for this module), returns an empty dict.

        Args:
            module_name (str): The name of the module for which to retrieve
              parameters.

        Returns:
            A dict mapping parameter names to parameter values.
            Example: {"pdgTypes": [211], "nTracks": 1}
        """
        try:
            return self.module_params[module_name]
        except:
            return dict()

    def send_data(self, df, key="data"):
        """Sends event data to the client.

        Creates an in-memory store from the given DataFrame which is used
        to obtain the binarized representation of the data. The binary data
        is then sent to the client. Note that the server must send the 
        data as a reply to a client's request, so this method will block 
        indefinitely if the client does not send a "ready" message.

        Args:
            df (pd.DataFrame): A DataFrame containing the harvested 
              event data.
            key (str): The key for the H5 store. Needed for decoding
              the binary data on the client's end. (Defaults to "data".)
        """
        # Create an in-memory store
        store = pd.HDFStore(
            f"{str(uuid.uuid4())}.h5",
            driver="H5FD_CORE",
            driver_core_backing_store=0
        )
        store[key] = df

        # Get binarized data
        binmsg = store._handle.get_file_image()
        store.close()

        # Send binary data
        message = self.socket.recv_string()
        logging.debug(f"Received {message}.")
        if message != "ready":
            print(f"Error: client message: {message}.")
            return
        self.socket.send(binmsg)


class DataCollector(b2.Module):
    """A simple data harvesting class.
    
    This class inherits the BASF2 Module class and must be registered on 
    the BASF2 path. Its role is to harvest event data from StoreArrays and 
    various BASF2 objects. The get_data() method performs the data
    harvesting, and is called by the module's event() method, which 
    handles sending the data to the client. Note: because of the ZMQ 
    server's request/reply structure, the DataCollector *must* send data to 
    the client.

    Attributes:
        server (Server): A reference to the script's Server object,
          maintained in order to send data back to the client.
    """

    def __init__(self, server):
        """Inits DataCollector with reference to server.

        Args:
            server (Server): A reference to the Server object.
        """
        
        super().__init__()
        self.__name__ = 'Data'
        self.server = server

    def get_data(self):
        """Harvests data from the event.

        Design your data harvesting HERE. The given example gets
        information about the Cherenkov photons detected in the TOP for
        the given event. The harvested data must be compiled into a 
        DataFrame and returned.

        Returns:
            A pd.DataFrame object containing the data. If there is no
            data to be collected for this specific event, return an
            empty DataFrame. (See the `if not mcps/track/logls` checks
            lines in the example implementation.)
        """
        
        data = []
        col_names = [
            'channelID', 
            'detectTime', 
            'emitX', 
            'emitY', 
            'emitZ', 
            'emitTime', 
            'moduleID'
        ]

        print('n digits:', len(Belle2.PyStoreArray("TOPDigits")))
        print('n barhits:', len(Belle2.PyStoreArray("TOPBarHits")))
        print('n mcps:', len(Belle2.PyStoreArray("MCParticles")))
        print('n simhits:', len(Belle2.PyStoreArray("TOPSimHits")))
        
        mcps = Belle2.PyStoreArray("MCParticles")
        if not mcps or len(mcps) == 0:
            print('no mcps')
            return pd.DataFrame()
        mcp = sorted(mcps, key=lambda x: x.getEnergy(), reverse=True)[0]
        track = mcp.getRelated('Tracks')
        if not track:
            print('no track')
            return pd.DataFrame()
        
        digits = Belle2.PyStoreArray('TOPDigits')
        if not digits:
            print('no digits')
            return pd.DataFrame()

        # Get the photons' info and add to the data list
        for d in digits:
            simhit = d.getRelated('TOPSimHits')
            if not simhit:
                continue
            photon = simhit.getRelated('TOPSimPhotons')
            if not photon:
                continue

            moduleID = d.getModuleID()
            channelID = d.getPixelID() - 1
            time = photon.getDetectionTime()
            point = photon.getEmissionPoint()
            eTime = photon.getEmissionTime()

            data.append(
                (
                    channelID,
                    time,
                    point.X(), 
                    point.Y(), 
                    point.Z(),
                    eTime,
                    moduleID
                )
            )

        # Now we return the data in a DataFrame
        return pd.DataFrame(data=data, columns=col_names)

    def event(self):
        """Calls data harvesting function and sends data to client.

        Uses the self.get_data() function to get a DataFrame containing
        the event data, unless the terminate command has been received 
        by the server (in which case an empty DataFrame is sent). Note
        that send_df() will block indefinitely until it receives a request
        from the client (see Server.send_df() docs for more).
        """

        if self.server.shut_down:
            logging.debug("Terminate active: sending empty DataFrame")
            self.server.send_data(pd.DataFrame())
            return
        
        out_df = self.get_data()
        logging.debug("Sending DataFrame")
        self.server.send_data(out_df)

    def terminate(self):
        """Functionally empty terminate method."""
        logging.debug("Terminating")


class Wrapper(b2.Module):
    """Wrapper class for a varying module.
    
    This class is designed to be a lightweight wrapper around any modules
    whose parameters will vary between events. For each event, it gets
    and sets the module's parameters before calling the module's event()
    method.

    Attributes:
        name (str): The name of the wrapped module.
        module: The actual module object, registered with BASF2.
        server (Server): The script's Server object.
    """

    def __init__(self, name, server):
        """Inits the Wrapper module.

        Args:
            name (str): The name of the module to wrap.
            server (Server): A reference to the script's Server object.
        """
        super().__init__()
        self.name = name
        self.module = b2.register_module(name)
        self.server = server

    def initialize(self):
        """Calls module's initialize() method."""
        self.module.initialize()

    def event(self):
        """Calls module's event() method after setting new parameters."""
        params = self.server.get_params(self.name)
        for key, val in params.items():
            self.module.param(key, val)
        self.module.event()

    def terminate(self):
        """Calls module's termiante() method."""
        self.module.terminate()


if __name__ == "__main__":
    
    p = b2.Path()

    eventinfosetter = b2.register_module('EventInfoSetter')
    eventinfosetter.param('evtNumList', [2])
    eventinfosetter.param('resetEvery', True) # allows indefinite # events
    p.add_module(eventinfosetter)
    
    server = Server(eventinfosetter)
    p.add_module(server)

    gearbox = b2.register_module('Gearbox')
    p.add_module(gearbox)
    
    geometry = b2.register_module('Geometry')
    p.add_module(geometry)
    
    # SimulationServiceInput module in lieu of varying module(s)
    particlegun = Wrapper('ParticleGun', server)
    p.add_module(particlegun)

    simulation = b2.register_module('FullSim')
    simulation.param('PhotonFraction', 1.0)
    p.add_module(simulation)
    
    topdigi = b2.register_module('TOPDigitizer')
    topdigi.param('useWaveforms', False)
    topdigi.param('simulateTTS', False)
    topdigi.param('electronicJitter', 0.0)
    topdigi.param('timeZeroJitter', 0.0)
    p.add_module(topdigi)
    
    trackmaker = b2.register_module('TOPMCTrackMaker')
    p.add_module(trackmaker)
    
    topreco = b2.register_module('TOPReconstructor')
    p.add_module(topreco)
    
    # Data collecting module at end!
    d = DataCollector(server)
    p.add_module(d)
    
    progress = b2.register_module('Progress')
    p.add_module(progress)
    
    b2.process(p)
