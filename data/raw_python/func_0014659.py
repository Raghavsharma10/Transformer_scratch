def getElementsByClassName(self, className, root='root', useIndex=True):
        '''
            getElementsByClassName - Searches and returns all elements containing a given class name.
               
                @param className <str> - A one-word class name
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
                @param useIndex <bool> If useIndex is True and class names are indexed [see constructor] only the index will be used. Otherwise a full search is performed.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        if useIndex is True and self.indexClassNames is True:

            elements = self._classNameMap.get(className, [])

            if isFromRoot is False:
                _hasTagInParentLine = self._hasTagInParentLine
                elements = [x for x in elements if _hasTagInParentLine(x, root)]

            return TagCollection(elements)

        return AdvancedHTMLParser.getElementsByClassName(self, className, root)