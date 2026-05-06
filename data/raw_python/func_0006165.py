def get_pool_for_host(self, host_id):
        """Returns the connection pool for the given host.

        This connection pool is used by the redis clients to make sure
        that it does not have to reconnect constantly.  If you want to use
        a custom redis client you can pass this in as connection pool
        manually.
        """
        if isinstance(host_id, HostInfo):
            host_info = host_id
            host_id = host_info.host_id
        else:
            host_info = self.hosts.get(host_id)
            if host_info is None:
                raise LookupError('Host %r does not exist' % (host_id,))

        rv = self._pools.get(host_id)
        if rv is not None:
            return rv
        with self._lock:
            rv = self._pools.get(host_id)
            if rv is None:
                opts = dict(self.pool_options or ())
                opts['db'] = host_info.db
                opts['password'] = host_info.password
                if host_info.unix_socket_path is not None:
                    opts['path'] = host_info.unix_socket_path
                    opts['connection_class'] = UnixDomainSocketConnection
                    if host_info.ssl:
                        raise TypeError('SSL is not supported for unix '
                                        'domain sockets.')
                else:
                    opts['host'] = host_info.host
                    opts['port'] = host_info.port
                    if host_info.ssl:
                        if SSLConnection is None:
                            raise TypeError('This version of py-redis does '
                                            'not support SSL connections.')
                        opts['connection_class'] = SSLConnection
                        opts.update(('ssl_' + k, v) for k, v in
                                    (host_info.ssl_options or {}).iteritems())
                rv = self.pool_cls(**opts)
                self._pools[host_id] = rv
            return rv