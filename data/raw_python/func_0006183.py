def fanout(self, hosts=None, timeout=None, max_concurrency=64,
               auto_batch=None):
        """Returns a context manager for a map operation that fans out to
        manually specified hosts instead of using the routing system.  This
        can for instance be used to empty the database on all hosts.  The
        context manager returns a :class:`FanoutClient`.  Example usage::

            with cluster.fanout(hosts=[0, 1, 2, 3]) as client:
                results = client.info()
            for host_id, info in results.value.iteritems():
                print '%s -> %s' % (host_id, info['is'])

        The promise returned accumulates all results in a dictionary keyed
        by the `host_id`.

        The `hosts` parameter is a list of `host_id`\s or alternatively the
        string ``'all'`` to send the commands to all hosts.

        The fanout APi needs to be used with a lot of care as it can cause
        a lot of damage when keys are written to hosts that do not expect
        them.
        """
        return MapManager(self.get_fanout_client(hosts, max_concurrency,
                                                 auto_batch),
                          timeout=timeout)