def all(self, timeout=None, max_concurrency=64, auto_batch=True):
        """Fanout to all hosts.  Works otherwise exactly like :meth:`fanout`.

        Example::

            with cluster.all() as client:
                client.flushdb()
        """
        return self.fanout('all', timeout=timeout,
                           max_concurrency=max_concurrency,
                           auto_batch=auto_batch)