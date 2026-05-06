def getElementsByAttr(self, attrName, attrValue, root='root'):
        '''
            getElementsByAttr - Searches the full tree for elements with a given attribute name and value combination. This is always a full scan.
               
                @param attrName <lowercase str> - A lowercase attribute name
                @param attrValue <str> - Expected value of attribute
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root', the root of the parsed tree will be used.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []

        if isFromRoot is True and root.getAttribute(attrName) == attrValue:
            elements.append(root)

        getElementsByAttr = self.getElementsByAttr
        for child in root.children:

            if child.getAttribute(attrName) == attrValue:
                elements.append(child)

            elements += getElementsByAttr(attrName, attrValue, child)

        return TagCollection(elements)