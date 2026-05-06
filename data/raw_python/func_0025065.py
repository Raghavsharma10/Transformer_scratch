def item_transaction(self, item) -> Transaction:
        """Begin transaction state for item.

        A transaction state is exists to prevent writing out to disk, mainly for performance reasons.
        All changes to the object are delayed until the transaction state exits.

        This method is thread safe.
        """
        items = self.__build_transaction_items(item)
        transaction = Transaction(self, item, items)
        self.__transactions.append(transaction)
        return transaction