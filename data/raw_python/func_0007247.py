def holes(self):
        '''Return holes in collection.

        Return :py:class:`~clique.collection.Collection` of missing indexes.

        '''
        missing = set([])
        previous = None
        for index in self.indexes:
            if previous is None:
                previous = index
                continue

            if index != (previous + 1):
                missing.update(range(previous + 1, index))

            previous = index

        return Collection(self.head, self.tail, self.padding, indexes=missing)