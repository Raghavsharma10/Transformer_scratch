def disconnect(self):
        """Disconnects the object.

        Safe method (no exception, even if it's already disconnected or if
        there are some connection errors).
        """
        if not self.is_connected() and not self.is_connecting():
            return
        LOG.debug("disconnecting from %s...", self._redis_server())
        self.__periodic_callback.stop()
        try:
            self._ioloop.remove_handler(self.__socket_fileno)
            self._listened_events = 0
        except Exception:
            pass
        self.__socket_fileno = -1
        try:
            self.__socket.close()
        except Exception:
            pass
        self._state.set_disconnected()
        self._close_callback()
        LOG.debug("disconnected from %s", self._redis_server())