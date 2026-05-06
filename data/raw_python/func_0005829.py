def set_zerg_client_params(self, server_sockets, use_fallback_socket=None):
        """Zerg mode. Zergs params.

        :param str|unicode|list[str|unicode] server_sockets: Attaches zerg to a zerg server.

        :param bool use_fallback_socket: Fallback to normal sockets if the zerg server is not available

        """
        self._set('zerg', server_sockets, multi=True)

        if use_fallback_socket is not None:
            self._set('zerg-fallback', use_fallback_socket, cast=bool)

            for socket in listify(server_sockets):
                self._section.networking.register_socket(self._section.networking.sockets.default(socket))

        return self._section