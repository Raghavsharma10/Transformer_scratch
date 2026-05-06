def getElementsWithAttrValues(self, attrName, values, root='root', useIndex=True):
        '''
            getElementsWithAttrValues - Returns elements with an attribute matching one of several values. For a single name/value combination, see getElementsByAttr

                @param attrName <lowercase str> - A lowercase attribute name
                @param attrValues set<str> - List of expected values of attribute
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
                @param useIndex <bool> If useIndex is True and this specific attribute is indexed [see addIndexOnAttribute] only the index will be used. Otherwise a full search is performed.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        _otherAttributeIndexes = self._otherAttributeIndexes
        if useIndex is True and attrName in _otherAttributeIndexes:

            elements = TagCollection()

            for value in values:
                elements += TagCollection(_otherAttributeIndexes[attrName].get(value, []))

            return elements

        return AdvancedHTMLParser.getElementsWithAttrValues(self, attrName, values, root, useIndex)