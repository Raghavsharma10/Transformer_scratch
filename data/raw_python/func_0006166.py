def map(self, timeout=None, max_concurrency=64, auto_batch=True):
        """Shortcut context manager for getting a routing client, beginning
        a map operation and joining over the result.  `max_concurrency`
        defines how many outstanding parallel queries can exist before an
        implicit join takes place.

        In the context manager the client available is a
        :class:`MappingClient`.  Example usage::

            results = {}
            with cluster.map() as client:
                for key in keys_to_fetch:
                    results[key] = client.get(key)
            for key, promise in results.iteritems():
                print '%s => %s' % (key, promise.value)
        """
        return self.get_routing_client(auto_batch).map(
            timeout=timeout, max_concurrency=max_concurrency)