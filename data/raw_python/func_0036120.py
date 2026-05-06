def range_by_score(self, min, max, num=None, withscores=False):
        """Return all the elements with score >= min and score <= max
        (a range query) from the sorted set."""
        return self.client.zrangebyscore(self.name, min, max, num=num,
                                         withscores=withscores)