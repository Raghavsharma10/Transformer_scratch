def transaction(self, mode="w"):
        """
        Create a new :class:`~bloop.transactions.ReadTransaction` or :class:`~bloop.transactions.WriteTransaction`.

        As a context manager, calling commit when the block exits:

        .. code-block:: pycon

            >>> engine = Engine()
            >>> user = User(id=3, email="user@domain.com")
            >>> tweet = Tweet(id=42, data="hello, world")
            >>> with engine.transaction("w") as tx:
            ...     tx.delete(user)
            ...     tx.save(tweet, condition=Tweet.id.is_(None))

        Or manually calling prepare and commit:

        .. code-block:: pycon

            >>> engine = Engine()
            >>> user = User(id=3, email="user@domain.com")
            >>> tweet = Tweet(id=42, data="hello, world")
            >>> tx = engine.transaction("w")
            >>> tx.delete(user)
            >>> tx.save(tweet, condition=Tweet.id.is_(None))
            >>> tx.prepare().commit()

        :param str mode: Either "r" or "w" to create a ReadTransaction or WriteTransaction.  Default is "w"
        :return: A new transaction that can be committed.
        :rtype: :class:`~bloop.transactions.ReadTransaction` or :class:`~bloop.transactions.WriteTransaction`
        """
        if mode == "r":
            cls = ReadTransaction
        elif mode == "w":
            cls = WriteTransaction
        else:
            raise ValueError(f"unknown mode {mode}")
        return cls(self)