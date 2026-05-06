def connect(self):
        """Connects the object to the host:port.

        Returns:
            Future: a Future object with True as result if the connection
                process was ok.
        """
        if self.is_connected() or self.is_connecting():
            raise tornado.gen.Return(True)
        if self.unix_domain_socket is None:
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.tcp_nodelay:
                self.__socket.setsockopt(socket.IPPROTO_TCP,
                                         socket.TCP_NODELAY, 1)
        else:
            if not os.path.exists(self.unix_domain_socket):
                LOG.warning("can't connect to %s, file does not exist",
                            self.unix_domain_socket)
                raise tornado.gen.Return(False)
            self.__socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.__socket.setblocking(0)
        self.__periodic_callback.start()
        try:
            LOG.debug("connecting to %s...", self._redis_server())
            self._state.set_connecting()
            if self.unix_domain_socket is None:
                self.__socket.connect((self.host, self.port))
            else:
                self.__socket.connect(self.unix_domain_socket)
        except socket.error as e:
            if (errno_from_exception(e) not in _ERRNO_INPROGRESS and
                    errno_from_exception(e) not in _ERRNO_WOULDBLOCK):
                self.disconnect()
                LOG.warning("can't connect to %s", self._redis_server())
                raise tornado.gen.Return(False)
            self.__socket_fileno = self.__socket.fileno()
            self._register_or_update_event_handler()
            yield self._state.get_changed_state_future()
            if not self.is_connected():
                LOG.warning("can't connect to %s", self._redis_server())
                raise tornado.gen.Return(False)
        else:
            LOG.debug("connected to %s", self._redis_server())
            self.__socket_fileno = self.__socket.fileno()
            self._state.set_connected()
            self._register_or_update_event_handler()
        raise tornado.gen.Return(True)