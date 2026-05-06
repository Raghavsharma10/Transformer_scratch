def send_buffer(self):
        """Utility function that sends the buffer into the provided socket.
        The buffer itself will slowly clear out and is modified in place.
        """
        buf = self._send_buf
        sock = self.connection._sock
        try:
            timeout = sock.gettimeout()
            sock.setblocking(False)
            try:
                for idx, item in enumerate(buf):
                    sent = 0
                    while 1:
                        try:
                            sent = sock.send(item)
                        except IOError as e:
                            if e.errno == errno.EAGAIN:
                                continue
                            elif e.errno == errno.EWOULDBLOCK:
                                break
                            raise
                        self.sent_something = True
                        break
                    if sent < len(item):
                        buf[:idx + 1] = [item[sent:]]
                        break
                else:
                    del buf[:]
            finally:
                sock.settimeout(timeout)
        except IOError as e:
            self.connection.disconnect()
            if isinstance(e, socket.timeout):
                raise TimeoutError('Timeout writing to socket (host %s)'
                                   % self.host_id)
            raise ConnectionError('Error while writing to socket (host %s): %s'
                                  % (self.host_id, e))