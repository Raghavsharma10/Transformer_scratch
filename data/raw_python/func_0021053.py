def items(
        self,
        min_rank=None,
        max_rank=None,
        min_score=None,
        max_score=None,
        reverse=False,
        pipe=None,
    ):
        """
        Return a list of ``(member, score)`` tuples whose ranking is between
        *min_rank* and *max_rank* AND whose score is between *min_score* and
        *max_score* (both ranges inclusive). If no bounds are specified, all
        items will be returned.
        """
        pipe = self.redis if pipe is None else pipe

        no_ranks = (min_rank is None) and (max_rank is None)
        no_scores = (min_score is None) and (max_score is None)

        # Default scope: everything
        if no_ranks and no_scores:
            ret = self.items_by_score(min_score, max_score, reverse, pipe)
        # Scope narrows to given score range
        elif no_ranks and (not no_scores):
            ret = self.items_by_score(min_score, max_score, reverse, pipe)
        # Scope narrows to given rank range
        elif (not no_ranks) and no_scores:
            ret = self.items_by_rank(min_rank, max_rank, reverse, pipe)
        # Scope narrows twice - once by rank and once by score
        else:
            results = self.items_by_rank(min_rank, max_rank, reverse, pipe)
            ret = []
            for member, score in results:
                if (min_score is not None) and (score < min_score):
                    continue
                if (max_score is not None) and (score > max_score):
                    continue
                ret.append((member, score))

        return ret