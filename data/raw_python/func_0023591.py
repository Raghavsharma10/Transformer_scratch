def close(self):
        '''Close our connection'''
        # Flush any unsent message
        try:
            while self.pending():
                self.flush()
        except socket.error:
            pass
        with self._socket_lock:
            try:
                if self._socket:
                    self._socket.close()
            finally:
                self._reset()