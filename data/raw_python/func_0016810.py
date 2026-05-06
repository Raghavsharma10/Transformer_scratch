def save(self, *objs, condition=None, atomic=False) -> "WriteTransaction":
        """
        Add one or more objects to be saved in this transaction.

        At most 10 items can be checked, saved, or deleted in the same transaction.  The same idempotency token will
        be used for a single prepared transaction, which allows you to safely call commit on the PreparedCommit object
        multiple times.

        :param objs: Objects to add to the set that are updated in this transaction.
        :param condition: A condition for these objects which must hold for the transaction to commit.
        :param bool atomic: only commit the transaction if the local and DynamoDB versions of the object match.
        :return: this transaction for chaining
        """
        self._extend([TxItem.new("save", obj, condition, atomic) for obj in objs])
        return self