def disableIndexing(self):
        '''
            disableIndexing - Disables indexing. Consider using plain AdvancedHTMLParser class.
              Maybe useful in some scenarios where you want to parse, add a ton of elements, then index
              and do a bunch of searching.
        '''
        self.indexIDs = self.indexNames = self.indexClassNames = self.indexTagNames = False
        self._resetIndexInternal()