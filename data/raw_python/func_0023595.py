def flush(self):
        '''Flush some of the waiting messages, returns count written'''
        # When profiling, we found that while there was some efficiency to be
        # gained elsewhere, the big performance hit is sending lots of small
        # messages at a time. In particular, consumers send many 'FIN' messages
        # which are very small indeed and the cost of dispatching so many system
        # calls is very high. Instead, we prefer to glom together many messages
        # into a single string to send at once.
        total = 0
        for sock in self.socket(blocking=False):
            # If there's nothing left in the out buffer, take whatever's in the
            # pending queue.
            #
            # When using SSL, if the socket throws 'SSL_WANT_WRITE', then the
            # subsequent send requests have to send the same buffer.
            pending = self._pending
            data = self._out_buffer or ''.join(
                pending.popleft() for _ in xrange(len(pending)))
            try:
                # Try to send as much of the first message as possible
                total = sock.send(data)
            except socket.error as exc:
                # Catch (errno, message)-type socket.errors
                if exc.args[0] not in self.WOULD_BLOCK_ERRS:
                    raise
                self._out_buffer = data
            else:
                self._out_buffer = None
            finally:
                if total < len(data):
                    # Save the rest of the message that could not be sent
                    self._pending.appendleft(data[total:])
        return total