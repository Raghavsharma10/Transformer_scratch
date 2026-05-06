def set_bsd_socket_params(self, port_reuse=None):
        """Sets BSD-sockets related params.

        :param bool port_reuse: Enable REUSE_PORT flag on socket to allow multiple
            instances binding on the same address (BSD only).

        """
        self._set('reuse-port', port_reuse, cast=bool)

        return self._section