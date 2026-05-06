def addIndexOnAttribute(self, attributeName):
        '''
            addIndexOnAttribute - Add an index for an arbitrary attribute. This will be used by the getElementsByAttr function.
                You should do this prior to parsing, or call reindex. Otherwise it will be blank. "name" and "id" will have no effect.
    
                @param attributeName <lowercase str> - An attribute name. Will be lowercased.
        '''
        attributeName = attributeName.lower()
        self._otherAttributeIndexes[attributeName] = {}

        def _otherIndexFunction(self, tag):
            thisAttribute = tag.getAttribute(attributeName)
            if thisAttribute is not None:
                if thisAttribute not in self._otherAttributeIndexes[attributeName]:
                    self._otherAttributeIndexes[attributeName][thisAttribute] = []
                self._otherAttributeIndexes[attributeName][thisAttribute].append(tag)


        self.otherAttributeIndexFunctions[attributeName] = _otherIndexFunction