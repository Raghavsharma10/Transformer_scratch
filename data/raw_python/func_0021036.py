def isdisjoint(self, other):
        """
        Return ``True`` if the set has no elements in common with *other*.
        Sets are disjoint if and only if their intersection is the empty set.

        :param other: Any kind of iterable.
        :rtype: boolean
        """
        def isdisjoint_trans_pure(pipe):
            return not pipe.sinter(self.key, other.key)

        def isdisjoint_trans_mixed(pipe):
            self_values = set(self.__iter__(pipe))
            if use_redis:
                other_values = set(other.__iter__(pipe))
            else:
                other_values = set(other)

            return self_values.isdisjoint(other_values)

        if self._same_redis(other):
            return self._transaction(isdisjoint_trans_pure, other.key)
        if self._same_redis(other, RedisCollection):
            use_redis = True
            return self._transaction(isdisjoint_trans_mixed, other.key)

        use_redis = False
        return self._transaction(isdisjoint_trans_mixed)