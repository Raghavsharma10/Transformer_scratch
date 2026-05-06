def reset(self):
        """ Add a RESET message to the outgoing queue, send
        it and consume all remaining messages.
        """

        def fail(metadata):
            raise ProtocolError("RESET failed %r" % metadata)

        log_debug("[#%04X]  C: RESET", self.local_port)
        self._append(b"\x0F", response=Response(self, on_failure=fail))
        self.sync()