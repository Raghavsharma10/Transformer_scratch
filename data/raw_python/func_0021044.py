def discard_member(self, member, pipe=None):
        """
        Remove *member* from the collection, unconditionally.
        """
        pipe = self.redis if pipe is None else pipe
        pipe.zrem(self.key, self._pickle(member))