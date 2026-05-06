def update(self, other):
        """
        Update the collection with items from *other*. Accepts other
        :class:`SortedSetBase` instances, dictionaries mapping members to
        numeric scores, or sequences of ``(member, score)`` tuples.
        """
        def update_trans(pipe):
            other_items = method(pipe=pipe) if use_redis else method()

            pipe.multi()
            for member, score in other_items:
                pipe.zadd(self.key, {self._pickle(member): float(score)})

        watches = []
        if self._same_redis(other, RedisCollection):
            use_redis = True
            watches.append(other.key)
        else:
            use_redis = False

        if hasattr(other, 'items'):
            method = other.items
        elif hasattr(other, '__iter__'):
            method = other.__iter__

        self._transaction(update_trans, *watches)