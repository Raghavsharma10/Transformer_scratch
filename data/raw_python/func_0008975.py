def get_connected_client(self):
        """Gets a connected Client object.

        If max_size is reached, this method will block until a new client
        object is available.

        Returns:
            A Future object with connected Client instance as a result
                (or ClientError if there was a connection problem)
        """
        if self.__sem is not None:
            yield self.__sem.acquire()
        client = None
        newly_created, client = self._get_client_from_pool_or_make_it()
        if newly_created:
            res = yield client.connect()
            if not res:
                LOG.warning("can't connect to %s", client.title)
                raise tornado.gen.Return(
                    ClientError("can't connect to %s" % client.title))
        raise tornado.gen.Return(client)