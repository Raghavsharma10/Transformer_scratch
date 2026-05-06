def map(self, timeout=None, max_concurrency=64, auto_batch=None):
        """Returns a context manager for a map operation.  This runs
        multiple queries in parallel and then joins in the end to collect
        all results.

        In the context manager the client available is a
        :class:`MappingClient`.  Example usage::

            results = {}
            with cluster.map() as client:
                for key in keys_to_fetch:
                    results[key] = client.get(key)
            for key, promise in results.iteritems():
                print '%s => %s' % (key, promise.value)
        """
        return MapManager(self.get_mapping_client(max_concurrency, auto_batch),
                          timeout=timeout)