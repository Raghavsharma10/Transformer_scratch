def set_unix_socket_params(self, abstract=None, permissions=None, owner=None, umask=None):
        """Sets Unix-socket related params.

        :param bool abstract: Force UNIX socket into abstract mode (Linux only).

        :param str permissions: UNIX sockets are filesystem objects that obey
            UNIX permissions like any other filesystem object.

            You can set the UNIX sockets' permissions with this option if your webserver
            would otherwise have no access to the uWSGI socket. When used without a parameter,
            the permissions will be set to 666. Otherwise the specified chmod value will be used.

        :param str|unicode owner: Chown UNIX sockets.

        :param str|unicode umask: Set UNIX socket umask.

        """
        self._set('abstract-socket', abstract, cast=bool)
        self._set('chmod-socket', permissions)
        self._set('chown-socket', owner)
        self._set('umask', umask)

        return self._section