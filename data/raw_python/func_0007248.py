def is_compatible(self, collection):
        '''Return whether *collection* is compatible with this collection.

        To be compatible *collection* must have the same head, tail and padding
        properties as this collection.

        '''
        return all([
            isinstance(collection, Collection),
            collection.head == self.head,
            collection.tail == self.tail,
            collection.padding == self.padding
        ])