def getElementsByClassName(self, className, root='root'):
        '''
            getElementsByClassName - Searches and returns all elements containing a given class name.
               
                @param className <str> - A one-word class name
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root' [default], the root of the parsed tree will be used.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []

        if isFromRoot is True and className in root.classNames:
            elements.append(root)

        getElementsByClassName = self.getElementsByClassName
        for child in root.children:

            if className in child.classNames:
                elements.append(child)

            elements += getElementsByClassName(className, child)

        return TagCollection(elements)