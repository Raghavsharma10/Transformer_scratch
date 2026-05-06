def daemonize(self, log_into, after_app_loading=False):
        """Daemonize uWSGI.

        :param str|unicode log_into: Logging destination:

            * File: /tmp/mylog.log

            * UPD: 192.168.1.2:1717

                .. note:: This will require an UDP server to manage log messages.
                    Use ``networking.register_socket('192.168.1.2:1717, type=networking.SOCK_UDP)``
                    to start uWSGI UDP server.

        :param str|unicode bool after_app_loading: Whether to daemonize after
            or before applications loading.

        """
        self._set('daemonize2' if after_app_loading else 'daemonize', log_into)

        return self._section