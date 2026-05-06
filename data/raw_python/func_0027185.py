def release(self, conn):
        """Revert back connection to pool."""
        if conn.in_transaction:
            raise InvalidRequestError(
                "Cannot release a connection with "
                "not finished transaction"
            )
        raw = conn.connection
        res = yield from self._pool.release(raw)
        return res