def connect(self):
        """Connects the client object to redis.

        It's safe to use this method even if you are already connected.
        Note: this method is useless with autoconnect mode (default).

        Returns:
            a Future object with True as result if the connection was ok.
        """
        if self.is_connected():
            raise tornado.gen.Return(True)
        cb1 = self._read_callback
        cb2 = self._close_callback
        self.__callback_queue = collections.deque()
        self._reply_list = []
        self.__reader = hiredis.Reader(replyError=ClientError)
        kwargs = self.connection_kwargs
        self.__connection = Connection(cb1, cb2, **kwargs)
        connection_status = yield self.__connection.connect()
        if connection_status is not True:
            # nothing left to do here, return
            raise tornado.gen.Return(False)
        if self.password is not None:
            authentication_status = yield self._call('AUTH', self.password)
            if authentication_status != b'OK':
                # incorrect password, return back the result
                LOG.warning("impossible to connect: bad password")
                self.__connection.disconnect()
                raise tornado.gen.Return(False)
        if self.db != 0:
            db_status = yield self._call('SELECT', self.db)
            if db_status != b'OK':
                LOG.warning("can't select db %s", self.db)
                raise tornado.gen.Return(False)
        raise tornado.gen.Return(True)