def discard_between(
        self,
        min_rank=None,
        max_rank=None,
        min_score=None,
        max_score=None,
    ):
        """
        Remove members whose ranking is between *min_rank* and *max_rank*
        OR whose score is between *min_score* and *max_score* (both ranges
        inclusive). If no bounds are specified, no members will be removed.
        """
        no_ranks = (min_rank is None) and (max_rank is None)
        no_scores = (min_score is None) and (max_score is None)

        # Default scope: nothing
        if no_ranks and no_scores:
            return

        # Scope widens to given score range
        if no_ranks and (not no_scores):
            return self.discard_by_score(min_score, max_score)

        # Scope widens to given rank range
        if (not no_ranks) and no_scores:
            return self.discard_by_rank(min_rank, max_rank)

        # Scope widens to score range and then rank range
        with self.redis.pipeline() as pipe:
            self.discard_by_score(min_score, max_score, pipe)
            self.discard_by_rank(min_rank, max_rank, pipe)
            pipe.execute()