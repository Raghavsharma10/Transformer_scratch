def socket(self, blocking=True):
        '''Blockingly yield the socket'''
        # If the socket is available, then yield it. Otherwise, yield nothing
        if self._socket_lock.acquire(blocking):
            try:
                yield self._socket
            finally:
                self._socket_lock.release()