def pfmerge(self, dest, *sources):
        """
        Merge N different HyperLogLogs into a single one.

        :param dest:
        :param sources:
        :return:
        """
        sources = [self.redis_key(k) for k in sources]
        with self.pipe as pipe:
            return pipe.pfmerge(self.redis_key(dest), *sources)