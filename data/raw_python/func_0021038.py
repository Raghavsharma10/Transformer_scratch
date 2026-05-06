def random_sample(self, k=1):
        """
        Return a *k* length list of unique elements chosen from the Set.
        Elements are not removed. Similar to :func:`random.sample` function
        from standard library.

        :param k: Size of the sample, defaults to 1.
        :rtype: :class:`list`

        """
        # k == 0: no work to do
        if k == 0:
            results = []
        # k == 1: same behavior on all versions of Redis
        elif k == 1:
            results = [self.redis.srandmember(self.key)]
        # k != 1, Redis version >= 2.6: compute in Redis
        else:
            results = self.redis.srandmember(self.key, k)

        return [self._unpickle(x) for x in results]