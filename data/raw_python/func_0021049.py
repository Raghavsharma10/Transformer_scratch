def get_score(self, member, default=None, pipe=None):
        """
        Return the score of *member*, or *default* if it is not in the
        collection.
        """
        pipe = self.redis if pipe is None else pipe
        score = pipe.zscore(self.key, self._pickle(member))

        if (score is None) and (default is not None):
            score = float(default)

        return score