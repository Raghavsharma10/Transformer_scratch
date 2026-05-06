def fanout(self, hosts=None, timeout=None, max_concurrency=64,
               auto_batch=True):
        """Shortcut context manager for getting a routing client, beginning
        a fanout operation and joining over the result.

        In the context manager the client available is a
        :class:`FanoutClient`.  Example usage::

            with cluster.fanout(hosts='all') as client:
                client.flushdb()
        """
        return self.get_routing_client(auto_batch).fanout(
            hosts=hosts, timeout=timeout, max_concurrency=max_concurrency)