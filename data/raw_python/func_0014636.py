def getElementsCustomFilter(self, filterFunc, root='root'):
        '''
            getElementsCustomFilter - Scan elements using a provided function

            @param filterFunc <function>(node) - A function that takes an AdvancedTag as an argument, and returns True if some arbitrary criteria is met

            @return - TagCollection of all matching elements
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []

        if isFromRoot is True and filterFunc(root) is True:
            elements.append(root)

        getElementsCustomFilter = self.getElementsCustomFilter
        for child in root.children:

            if filterFunc(child) is True:
                elements.append(child)

            elements += getElementsCustomFilter(filterFunc, child)

        return TagCollection(elements)