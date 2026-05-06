def subtract(self, other=None, **kwargs):
        """Elements are subtracted from an *iterable* or from another
        *mapping* (or counter). Like :func:`dict.update` but subtracts
        counts instead of replacing them.
        """
        if other is not None:
            if self._same_redis(other, RedisCollection):
                self._update_helper(other, operator.sub, use_redis=True)
            elif hasattr(other, 'keys'):
                self._update_helper(other, operator.sub)
            else:
                self._update_helper(collections.Counter(other), operator.sub)

        if kwargs:
            self._update_helper(kwargs, operator.sub)