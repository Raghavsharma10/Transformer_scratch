def merge(self, collection):
        '''Merge *collection* into this collection.

        If the *collection* is compatible with this collection then update
        indexes with all indexes in *collection*.

        raise :py:class:`~clique.error.CollectionError` if *collection* is not
        compatible with this collection.

        '''
        if not self.is_compatible(collection):
            raise clique.error.CollectionError('Collection is not compatible '
                                               'with this collection.')

        self.indexes.update(collection.indexes)