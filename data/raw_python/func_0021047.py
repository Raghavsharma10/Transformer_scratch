def count_between(self, min_score=None, max_score=None):
        """
        Returns the number of members whose score is between *min_score* and
        *max_score* (inclusive).
        """
        min_score = float('-inf') if min_score is None else float(min_score)
        max_score = float('inf') if max_score is None else float(max_score)

        return self.redis.zcount(self.key, min_score, max_score)