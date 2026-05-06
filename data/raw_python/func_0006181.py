def get_fanout_client(self, hosts, max_concurrency=64,
                          auto_batch=None):
        """Returns a thread unsafe fanout client.

        Returns an instance of :class:`FanoutClient`.
        """
        if auto_batch is None:
            auto_batch = self.auto_batch
        return FanoutClient(hosts, connection_pool=self.connection_pool,
                            max_concurrency=max_concurrency,
                            auto_batch=auto_batch)