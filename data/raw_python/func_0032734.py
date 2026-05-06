def zunionstore(self, dest, keys, aggregate=None):
        """
        Union multiple sorted sets specified by ``keys`` into
        a new sorted set, ``dest``. Scores in the destination will be
        aggregated based on the ``aggregate``, or SUM if none is provided.
        """
        with self.pipe as pipe:
            keys = [self.redis_key(k) for k in keys]
            return pipe.zunionstore(self.redis_key(dest), keys,
                                    aggregate=aggregate)