def _connect(self):
        """ Connect to the remote if not already connected. """
        if not self.connected.is_set():
            try:
                self.lock.acquire()
                # Another thread may have connected while we were
                # waiting to acquire the lock
                if not self.connected.is_set():
                    self._do_connect()
                    if self.keepalive:
                        self._transport.set_keepalive(self.keepalive)
                    self.connected.set()
            except GerritError:
                raise
            finally:
                self.lock.release()