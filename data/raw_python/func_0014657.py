def getElementsByName(self, name, root='root', useIndex=True):
        '''
            getElementsByName - Searches and returns all elements with a specific name.
               
                @param name <str> - A string of the name attribute
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
                @param useIndex <bool> If useIndex is True and names are indexed [see constructor] only the index will be used. Otherwise a full search is performed.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []
        if useIndex is True and self.indexNames is True:

            elements = self._nameMap.get(name, [])

            if isFromRoot is False:
                _hasTagInParentLine = self._hasTagInParentLine
                elements = [x for x in elements if _hasTagInParentLine(x, root)]

            return TagCollection(elements)

        return AdvancedHTMLParser.getElementsByName(self, name, root)