def put_many(self, items):  # pragma: no cover
        """Put many key-value pairs.

        This method may take advantage of performance or atomicity
        features of the underlying store. It does not guarantee that
        all items will be set in the same transaction, only that
        transactions may be used for performance.

        :param items: An iterable producing (key, value) tuples.

        """
        for key, value in items:
            self.put(key, value)