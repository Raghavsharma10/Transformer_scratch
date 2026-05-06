def close(self):
        """ Close the connection.
        """
        if not self._closed:
            if self.protocol_version >= 3:
                log_debug("[#%04X]  C: GOODBYE", self.local_port)
                self._append(b"\x02", ())
                try:
                    self.send()
                except ServiceUnavailable:
                    pass
            log_debug("[#%04X]  C: <CLOSE>", self.local_port)
            try:
                self.socket.close()
            except IOError:
                pass
            finally:
                self._closed = True