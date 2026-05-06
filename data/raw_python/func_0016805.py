def prepare(self):
        """
        Create a new PreparedTransaction that can be committed.

        This is called automatically when exiting the transaction as a context:

        .. code-block:: python

            >>> engine = Engine()
            >>> tx = WriteTransaction(engine)
            >>> prepared = tx.prepare()
            >>> prepared.commit()

            # automatically calls commit when exiting
            >>> with WriteTransaction(engine) as tx:
            ...     # modify the transaction here
            ...     pass
            >>> # tx commits here

        :return:
        """
        tx = PreparedTransaction()
        tx.prepare(
            engine=self.engine,
            mode=self.mode,
            items=self._items,
        )
        return tx