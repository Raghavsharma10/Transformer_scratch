def is_dirty(self) -> bool:
        """ Are there uncommitted changes? """
        if len(self._transactions) == 0:
            raise RuntimeError("is_dirty called outside a transaction.")
        if len(self._transactions[-1]) > 0:
            return True
        return False