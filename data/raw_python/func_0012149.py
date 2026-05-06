def reset(self):
        "Close the current failed connection and prepare for a new one"
        log.info("resetting client")
        rpc_client = self._rpc_client
        self._addrs.append(self._peer.addr)
        self.__init__(self._addrs)
        self._rpc_client = rpc_client
        self._dispatcher.rpc_client = rpc_client
        rpc_client._client = weakref.ref(self)