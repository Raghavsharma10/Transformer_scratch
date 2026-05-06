def reindex(self, newIndexIDs=None, newIndexNames=None, newIndexClassNames=None, newIndexTagNames=None):
        '''
            reindex - reindex the tree. Optionally, change what fields are indexed.

                @param newIndexIDs <bool/None>        - None to leave same, otherwise new value to index IDs
                @parma newIndexNames <bool/None>      - None to leave same, otherwise new value to index names
                @param newIndexClassNames <bool/None> - None to leave same, otherwise new value to index class names
                @param newIndexTagNames <bool/None>   - None to leave same, otherwise new value to index tag names
        '''
        if newIndexIDs is not None:
            self.indexIDs = newIndexIDs
        if newIndexNames is not None:
            self.indexNames = newIndexNames
        if newIndexClassNames is not None:
            self.newIndexClassNames = newIndexClassNames
        if newIndexTagNames is not None:
            self.newIndexTagNames = newIndexTagNames

        self._resetIndexInternal()
        self._indexTagRecursive(self.root)