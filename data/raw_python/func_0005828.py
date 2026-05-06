def set_zerg_server_params(self, socket, clients_socket_pool=None):
        """Zerg mode. Zerg server params.

        When your site load is variable, it would be nice to be able to add
        workers dynamically. Enabling Zerg mode you can allow zerg clients to attach
        to your already running server and help it in the work.

        * http://uwsgi-docs.readthedocs.io/en/latest/Zerg.html

        :param str|unicode socket: Unix socket to bind server to.

            Examples:
                * unix socket - ``/var/run/mutalisk``
                * Linux abstract namespace - ``@nydus``

        :param str|unicode|list[str|unicode] clients_socket_pool: This enables Zerg Pools.

            .. note:: Expects master process.

            Accepts sockets that will be mapped to Zerg socket.

            * http://uwsgi-docs.readthedocs.io/en/latest/Zerg.html#zerg-pools

        """
        if clients_socket_pool:
            self._set('zergpool', '%s:%s' % (socket, ','.join(listify(clients_socket_pool))), multi=True)

        else:
            self._set('zerg-server', socket)

        return self._section