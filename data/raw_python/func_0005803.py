def set_socket_params(
            self, send_timeout=None, keep_alive=None, no_defer_accept=None,
            buffer_send=None, buffer_receive=None):
        """Sets common socket params.

        :param int send_timeout: Send (write) timeout in seconds.

        :param bool keep_alive: Enable TCP KEEPALIVEs.

        :param bool no_defer_accept: Disable deferred ``accept()`` on sockets
            by default (where available) uWSGI will defer the accept() of requests until some data
            is sent by the client (this is a security/performance measure).
            If you want to disable this feature for some reason, specify this option.

        :param int buffer_send: Set SO_SNDBUF (bytes).

        :param int buffer_receive: Set SO_RCVBUF (bytes).

        """
        self._set('so-send-timeout', send_timeout)
        self._set('so-keepalive', keep_alive, cast=bool)
        self._set('no-defer-accept', no_defer_accept, cast=bool)
        self._set('socket-sndbuf', buffer_send)
        self._set('socket-rcvbuf', buffer_receive)

        return self._section