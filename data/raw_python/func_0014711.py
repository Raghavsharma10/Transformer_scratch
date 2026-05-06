def getElementsCustomFilter(self, filterFunc):
        '''
            getElementsCustomFilter - Searches children of this tag for those matching a provided user function

            @param filterFunc <function> - A function or lambda expression that should return "True" if the passed node matches criteria.

            @return - TagCollection of matching results

            @see getFirstElementCustomFilter
        '''
        elements = []

        for child in self.children:
            if filterFunc(child) is True:
                elements.append(child)
            elements += child.getElementsCustomFilter(filterFunc)

        return TagCollection(elements)