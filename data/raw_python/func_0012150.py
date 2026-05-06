def shutdown(self):
        'Close the hub connection'
        log.info("shutting down")
        self._peer.go_down(reconnect=False, expected=True)