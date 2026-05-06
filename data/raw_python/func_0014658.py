def getElementById(self, _id, root='root', useIndex=True):
        '''
            getElementById - Searches and returns the first (should only be one) element with the given ID.
               
                @param id <str> - A string of the id attribute.
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
                @param useIndex <bool> If useIndex is True and ids are indexed [see constructor] only the index will be used. Otherwise a full search is performed.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        if self.useIndex is True and self.indexIDs is True:

            element = self._idMap.get(_id, None)

            if isFromRoot is False and element is not None:

                if self._hasTagInParentLine(element, root) is False:
                    element = None
                    
            return element


        return AdvancedHTMLParser.getElementById(self, _id, root)