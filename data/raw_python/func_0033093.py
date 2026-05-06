def connect(self):
        """
        Creates the connection with the redis server.
        Return ``True`` if the connection works, else returns
        ``False``. It does not take any arguments.

        :return: ``Boolean`` value

        .. note::

           After creating the ``Queue`` object the user should call
           the ``connect`` method to create the connection.

        .. doctest::

           >>> from retask import Queue
           >>> q = Queue('test')
           >>> q.connect()
           True

        """
        config = self.config
        self.rdb = redis.Redis(config['host'], config['port'], config['db'],\
                              config['password'])
        try:
            info = self.rdb.info()
            self.connected = True
        except redis.ConnectionError:
            return False

        return True