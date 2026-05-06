def transaction_id(self):
        """
        Transaction ID for Transbank, a secure random int between 0 and 999999999.
        """
        if not self._transaction_id:
            self._transaction_id = random.randint(0, 10000000000 - 1)
        return self._transaction_id