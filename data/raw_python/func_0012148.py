def wait_connected(self, timeout=None):
        '''Wait for connections to be made and their handshakes to finish

        :param timeout:
            maximum time to wait in seconds. with None, there is no timeout.
        :type timeout: float or None

        :returns:
            ``True`` if all connections were made, ``False`` if one or more
            failed.
        '''
        result = self._peer.wait_connected(timeout)
        if not result:
            if timeout is not None:
                log.warn("connect wait timed out after %.2f seconds" % timeout)
        return result