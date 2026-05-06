def add(self, item):
        '''Add *item* to collection.

        raise :py:class:`~clique.error.CollectionError` if *item* cannot be
        added to the collection.

        '''
        match = self.match(item)
        if match is None:
            raise clique.error.CollectionError(
                'Item does not match collection expression.'
            )

        self.indexes.add(int(match.group('index')))