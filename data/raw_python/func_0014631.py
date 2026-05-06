def getElementsByName(self, name, root='root'):
        '''
            getElementsByName - Searches and returns all elements with a specific name.
               
                @param name <str> - A string of the name attribute
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root' [default], the root of the parsed tree will be used.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []

        if isFromRoot is True and root.name == name:
            elements.append(root)

        getElementsByName = self.getElementsByName
        for child in root.children:

            if child.getAttribute('name') == name:
                elements.append(child)

            elements += getElementsByName(name, child)

        return TagCollection(elements)