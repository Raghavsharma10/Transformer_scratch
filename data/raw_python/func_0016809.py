def check(self, obj, condition) -> "WriteTransaction":
        """
        Add a condition which must be met for the transaction to commit.

        While the condition is checked against the provided object, that object will not be modified.  It is only
        used to provide the hash and range key to apply the condition to.

        At most 10 items can be checked, saved, or deleted in the same transaction.  The same idempotency token will
        be used for a single prepared transaction, which allows you to safely call commit on the PreparedCommit object
        multiple times.


        :param obj: The object to use for the transaction condition.  This object will not be modified.
        :param condition: A condition on an object which must hold for the transaction to commit.
        :return: this transaction for chaining
        """
        self._extend([TxItem.new("check", obj, condition)])
        return self