def remove(self, item):
        '''Remove *item* from collection.

        raise :py:class:`~clique.error.CollectionError` if *item* cannot be
        removed from the collection.

        '''
        match = self.match(item)
        if match is None:
            raise clique.error.CollectionError(
                'Item not present in collection.'
            )

        index = int(match.group('index'))
        try:
            self.indexes.remove(index)
        except KeyError:
            raise clique.error.CollectionError(
                'Item not present in collection.'
            )