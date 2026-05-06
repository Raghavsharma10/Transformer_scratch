def extend(self, other):
        """
        Extend the right side of the the collection by appending values from
        the iterable *other*.
        """
        def extend_trans(pipe):
            values = list(other.__iter__(pipe)) if use_redis else other
            for v in values:
                self._append_helper(v, pipe)

        if self._same_redis(other, RedisCollection):
            use_redis = True
            self._transaction(extend_trans, other.key)
        else:
            use_redis = False
            self._transaction(extend_trans)