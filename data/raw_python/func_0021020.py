def _transaction(self, fn, *extra_keys):
        """Helper simplifying code within watched transaction.

        Takes *fn*, function treated as a transaction. Returns whatever
        *fn* returns. ``self.key`` is watched. *fn* takes *pipe* as the
        only argument.

        :param fn: Closure treated as a transaction.
        :type fn: function *fn(pipe)*
        :param extra_keys: Optional list of additional keys to watch.
        :type extra_keys: list
        :rtype: whatever *fn* returns
        """
        results = []

        def trans(pipe):
            results.append(fn(pipe))

        self.redis.transaction(trans, self.key, *extra_keys)
        return results[0]