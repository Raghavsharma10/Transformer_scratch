def remove(self, value):
        """Remove the first occurence of *value*."""
        def remove_trans(pipe):
            # If we're caching, we'll need to synchronize before removing.
            if self.writeback:
                self._sync_helper(pipe)

            delete_count = pipe.lrem(self.key, 1, self._pickle(value))
            if delete_count == 0:
                raise ValueError

        self._transaction(remove_trans)