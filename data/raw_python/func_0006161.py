def add_host(self, host_id=None, host='localhost', port=6379,
                 unix_socket_path=None, db=0, password=None,
                 ssl=False, ssl_options=None):
        """Adds a new host to the cluster.  This is only really useful for
        unittests as normally hosts are added through the constructor and
        changes after the cluster has been used for the first time are
        unlikely to make sense.
        """
        if host_id is None:
            raise RuntimeError('Host ID is required')
        elif not isinstance(host_id, (int, long)):
            raise ValueError('The host ID has to be an integer')
        host_id = int(host_id)
        with self._lock:
            if host_id in self.hosts:
                raise TypeError('Two hosts share the same host id (%r)' %
                                (host_id,))
            self.hosts[host_id] = HostInfo(host_id=host_id, host=host,
                                           port=port, db=db,
                                           unix_socket_path=unix_socket_path,
                                           password=password, ssl=ssl,
                                           ssl_options=ssl_options)
            self._hosts_age += 1