def extendleft(self, other):
        """
        Extend the left side of the the collection by appending values from
        the iterable *other*. Note that the appends will reverse the order
        of the given values.
        """
        def extendleft_trans(pipe):
            values = list(other.__iter__(pipe)) if use_redis else other
            for v in values:
                self._appendleft_helper(v, pipe)

        if self._same_redis(other, RedisCollection):
            use_redis = True
            self._transaction(extendleft_trans, other.key)
        else:
            use_redis = False
            self._transaction(extendleft_trans)