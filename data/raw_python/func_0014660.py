def getElementsByAttr(self, attrName, attrValue, root='root', useIndex=True):
        '''
            getElementsByAttr - Searches the full tree for elements with a given attribute name and value combination. If you want multiple potential values, see getElementsWithAttrValues
               If you want an index on a random attribute, use the addIndexOnAttribute function.
               
                @param attrName <lowercase str> - A lowercase attribute name
                @param attrValue <str> - Expected value of attribute
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
                @param useIndex <bool> If useIndex is True and this specific attribute is indexed [see addIndexOnAttribute] only the index will be used. Otherwise a full search is performed.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        if useIndex is True and attrName in self._otherAttributeIndexes:

            elements = self._otherAttributeIndexes[attrName].get(attrValue, [])

            if isFromRoot is False:
                _hasTagInParentLine = self._hasTagInParentLine
                elements = [x for x in elements if _hasTagInParentLine(x, root)]

            return TagCollection(elements)
        
        return AdvancedHTMLParser.getElementsByAttr(self, attrName, attrValue, root)