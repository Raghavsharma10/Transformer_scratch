def getFirstElementCustomFilter(self, filterFunc, root='root'):
        '''
            getFirstElementCustomFilter - Scan elements using a provided function, stop and return the first match.

                @see getElementsCustomFilter to match multiple elements

            @param filterFunc <function>(node) - A function that takes an AdvancedTag as an argument, and returns True if some arbitrary criteria is met

            @return - An AdvancedTag of the node that matched, or None if no match.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        elements = []

        if isFromRoot is True and filterFunc(root) is True:
            return root

        getFirstElementCustomFilter = self.getFirstElementCustomFilter

        for child in root.children:

            if filterFunc(child) is True:
                return child

            subRet = getFirstElementCustomFilter(filterFunc, child)

            if subRet:
                return subRet

        return None