def new_conn(self):
        """
        Create a new ConnectionWrapper instance
        :return:
        """
        """
        :return:
        """
        logger.debug("Opening new connection to rethinkdb with args=%s" % self._conn_args)
        return ConnectionWrapper(self._pool, **self._conn_args)