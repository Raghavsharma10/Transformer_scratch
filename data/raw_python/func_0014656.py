def getElementsByTagName(self, tagName, root='root', useIndex=True):
        '''
            getElementsByTagName - Searches and returns all elements with a specific tag name.
               
                @param tagName <lowercase str> - A lowercase string of the tag name. 
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
                @param useIndex - If True [default] and tag names are set to be indexed [default, see constructor], only the index will be used. If False, all tags
                  will be searched.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        if useIndex is True and self.indexTagNames is True:
            elements = self._tagNameMap.get(tagName, []) # Use .get here as to not create a lot of extra indexes on the defaultdict for misses
            if isFromRoot is False:
                _hasTagInParentLine = self._hasTagInParentLine
                elements = [x for x in elements if _hasTagInParentLine(x, root)]

            return TagCollection(elements)

        return AdvancedHTMLParser.getElementsByTagName(self, tagName, root)