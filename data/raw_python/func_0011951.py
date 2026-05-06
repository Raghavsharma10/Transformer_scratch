def start(self):
        "Start up the hub's server, and have it start initiating connections"
        log.info("starting")

        self._listener_coro = backend.greenlet(self._listener)
        self._udp_listener_coro = backend.greenlet(self._udp_listener)
        backend.schedule(self._listener_coro)
        backend.schedule(self._udp_listener_coro)

        for addr in self._peers:
            self.add_peer(addr)