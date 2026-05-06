def get_mapping_client(self, max_concurrency=64, auto_batch=None):
        """Returns a thread unsafe mapping client.  This client works
        similar to a redis pipeline and returns eventual result objects.
        It needs to be joined on to work properly.  Instead of using this
        directly you shold use the :meth:`map` context manager which
        automatically joins.

        Returns an instance of :class:`MappingClient`.
        """
        if auto_batch is None:
            auto_batch = self.auto_batch
        return MappingClient(connection_pool=self.connection_pool,
                             max_concurrency=max_concurrency,
                             auto_batch=auto_batch)