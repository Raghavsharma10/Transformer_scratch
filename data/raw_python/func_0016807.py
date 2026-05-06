def commit(self) -> None:
        """
        Commit the transaction with a fixed transaction id.

        A read transaction can call commit() any number of times, while a write transaction can only use the
        same tx_id for 10 minutes from the first call.
        """
        now = datetime.now(timezone.utc)
        if self.first_commit_at is None:
            self.first_commit_at = now

        if self.mode == "r":
            response = self.engine.session.transaction_read(self._request)
        elif self.mode == "w":
            if now - self.first_commit_at > MAX_TOKEN_LIFETIME:
                raise TransactionTokenExpired
            response = self.engine.session.transaction_write(self._request, self.tx_id)
        else:
            raise ValueError(f"unrecognized mode {self.mode}")

        self._handle_response(response)