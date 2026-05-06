def shutdown(self):
        'Close all peer connections and stop listening for new ones'
        log.info("shutting down")

        for peer in self._dispatcher.peers.values():
            peer.go_down(reconnect=False)

        if self._listener_coro:
            backend.schedule_exception(
                    errors._BailOutOfListener(), self._listener_coro)
        if self._udp_listener_coro:
            backend.schedule_exception(
                    errors._BailOutOfListener(), self._udp_listener_coro)