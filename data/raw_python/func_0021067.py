def extend(self, other):
        """
        Adds the values from the iterable *other* to the end of this
        collection.
        """
        def extend_trans(pipe):
            values = list(other.__iter__(pipe)) if use_redis else other
            len_self = pipe.rpush(self.key, *(self._pickle(v) for v in values))
            if self.writeback:
                for i, v in enumerate(values, len_self - len(values)):
                    self.cache[i] = v

        if self._same_redis(other, RedisCollection):
            use_redis = True
            self._transaction(extend_trans, other.key)
        else:
            use_redis = False
            self._transaction(extend_trans)