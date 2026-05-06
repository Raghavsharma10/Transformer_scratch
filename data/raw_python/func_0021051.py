def get_rank(self, member, reverse=False, pipe=None):
        """
        Return the rank of *member* in the collection.
        By default, the member with the lowest score has rank 0.
        If *reverse* is ``True``, the member with the highest score has rank 0.
        """
        pipe = self.redis if pipe is None else pipe
        method = getattr(pipe, 'zrevrank' if reverse else 'zrank')
        rank = method(self.key, self._pickle(member))

        return rank