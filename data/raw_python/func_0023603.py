def _read(self, limit=1000):
        '''Return all the responses read'''
        # It's important to know that it may return no responses or multiple
        # responses. It depends on how the buffering works out. First, read from
        # the socket
        for sock in self.socket():
            if sock is None:
                # Race condition. Connection has been closed.
                return []
            try:
                packet = sock.recv(4096)
            except socket.timeout:
                # If the socket times out, return nothing
                return []
            except socket.error as exc:
                # Catch (errno, message)-type socket.errors
                if exc.args[0] in self.WOULD_BLOCK_ERRS:
                    return []
                else:
                    raise

            # Append our newly-read data to our buffer
            self._buffer += packet

        responses = []
        total = 0
        buf = self._buffer
        remaining = len(buf)
        while limit and (remaining >= 4):
            size = struct.unpack('>l', buf[total:(total + 4)])[0]
            # Now check to see if there's enough left in the buffer to read
            # the message.
            if (remaining - 4) >= size:
                responses.append(Response.from_raw(
                    self, buf[(total + 4):(total + size + 4)]))
                total += (size + 4)
                remaining -= (size + 4)
                limit -= 1
            else:
                break
        self._buffer = self._buffer[total:]
        return responses