def reset(self, force_flush_cache: bool = False) -> None:
        """
        Reset transaction back to original state, discarding all
        uncompleted transactions.
        """
        super(LDAPwrapper, self).reset()
        if len(self._transactions) == 0:
            raise RuntimeError("reset called outside a transaction.")
        self._transactions[-1] = []