def get_or_set_score(self, member, default=0):
        """
        If *member* is in the collection, return its value. If not, store it
        with a score of *default* and return *default*. *default* defaults to
        0.
        """
        default = float(default)

        def get_or_set_score_trans(pipe):
            pickled_member = self._pickle(member)
            score = pipe.zscore(self.key, pickled_member)

            if score is None:
                pipe.zadd(self.key, {self._pickle(member): default})
                return default

            return score

        return self._transaction(get_or_set_score_trans)