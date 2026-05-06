def update(self, other=None, **kwargs):
        """Elements are counted from an *iterable* or added-in from another
        *mapping* (or counter). Like :func:`dict.update` but adds counts
        instead of replacing them. Also, the *iterable* is expected to be
        a sequence of elements, not a sequence of ``(key, value)`` pairs.
        """
        if other is not None:
            if self._same_redis(other, RedisCollection):
                self._update_helper(other, operator.add, use_redis=True)
            elif hasattr(other, 'keys'):
                self._update_helper(other, operator.add)
            else:
                self._update_helper(collections.Counter(other), operator.add)

        if kwargs:
            self._update_helper(kwargs, operator.add)