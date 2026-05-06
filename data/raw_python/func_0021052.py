def increment_score(self, member, amount=1):
        """
        Adjust the score of *member* by *amount*. If *member* is not in the
        collection it will be stored with a score of *amount*.
        """
        return self.redis.zincrby(
            self.key, float(amount), self._pickle(member)
        )