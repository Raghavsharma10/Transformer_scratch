def register_socket(self, socket):
        """Registers the given socket(s) for further use.

        :param Socket|list[Socket] socket: Socket type object. See ``.sockets``.

        """
        sockets = self._sockets

        for socket in listify(socket):

            uses_shared = isinstance(socket.address, SocketShared)

            if uses_shared:
                # Handling shared sockets involves socket index resolution.

                shared_socket = socket.address  # type: SocketShared

                if shared_socket not in sockets:
                    self.register_socket(shared_socket)

                socket.address = self._get_shared_socket_idx(shared_socket)

            socket.address = self._section.replace_placeholders(socket.address)
            self._set(socket.name, socket, multi=True)

            socket._contribute_to_opts(self)

            bound_workers = socket.bound_workers

            if bound_workers:
                self._set(
                    'map-socket', '%s:%s' % (len(sockets), ','.join(map(str, bound_workers))),
                    multi=True)

            if not uses_shared:
                sockets.append(socket)

        return self._section