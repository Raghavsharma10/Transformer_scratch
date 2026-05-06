def connect(self):
        "Initiate the connection to a proxying hub"
        log.info("connecting")

        # don't have the connection attempt reconnects, because when it goes
        # down we are going to cycle to the next potential peer from the Client
        self._peer = connection.Peer(
                None, self._dispatcher, self._addrs.popleft(),
                backend.Socket(), reconnect=False)
        self._peer.start()