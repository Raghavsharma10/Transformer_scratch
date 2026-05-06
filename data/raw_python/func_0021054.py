def set_score(self, member, score, pipe=None):
        """
        Set the score of *member* to *score*.
        """
        pipe = self.redis if pipe is None else pipe
        pipe.zadd(self.key, {self._pickle(member): float(score)})