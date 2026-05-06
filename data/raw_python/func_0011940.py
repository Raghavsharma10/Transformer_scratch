def wait_connected(self, conns=None, timeout=None):
        '''Wait for connections to be made and their handshakes to finish

        :param conns:
            a single or list of (host, port) tuples with the connections that
            must be finished before the method will return. defaults to all the
            peers the :class:`Hub` was instantiated with.
        :param timeout:
            maximum time to wait in seconds. with None, there is no timeout.
        :type timeout: float or None

        :returns:
            ``True`` if all connections were made, ``False`` one or more
            failed.
        '''
        if timeout:
            deadline = time.time() + timeout
        conns = conns or self._started_peers.keys()
        if not hasattr(conns, "__iter__"):
            conns = [conns]

        for peer_addr in conns:
            remaining = max(0, deadline - time.time()) if timeout else None
            if not self._started_peers[peer_addr].wait_connected(remaining):
                if timeout:
                    log.warn("connect wait timed out after %.2f seconds" %
                            timeout)
                return False
        return True