def heartbeat_timeout(self):
        """ Called by heartbeat_monitor on timeout """
        assert not self._closed, "Did we not stop heartbeat_monitor on close?"
        log.error("Heartbeat time out")
        poison_exc = ConnectionLostError('Heartbeat timed out')
        poison_frame = frames.PoisonPillFrame(poison_exc)
        self.dispatcher.dispatch_all(poison_frame)
        # Spec says to just close socket without ConnectionClose handshake.
        self.close()