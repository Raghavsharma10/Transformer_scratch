def find_one(self, **kwargs):
        """
        Works just like :py:meth:`find() <dataset.Table.find>` but returns one result, or None.
        ::
            row = table.find_one(country='United States')
        """
        kwargs["_limit"] = 1
        iterator = self.find(**kwargs)
        try:
            return next(iterator)
        except StopIteration:
            return None