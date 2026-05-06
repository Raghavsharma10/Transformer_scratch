def getElementsByTagName(self, tagName, root='root'):
        '''
            getElementsByTagName - Searches and returns all elements with a specific tag name.
               
                @param tagName <lowercase str> - A lowercase string of the tag name. 
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []

        if isFromRoot is True and root.tagName == tagName:
            elements.append(root)

        getElementsByTagName = self.getElementsByTagName
        for child in root.children:

            if child.tagName == tagName:
                elements.append(child)

            elements += getElementsByTagName(tagName, child)

        return TagCollection(elements)