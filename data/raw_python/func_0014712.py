def getFirstElementCustomFilter(self, filterFunc):
        '''
            getFirstElementCustomFilter - Gets the first element which matches a given filter func.

                Scans first child, to the bottom, then next child to the bottom, etc. Does not include "self" node.

            @param filterFunc <function> - A function or lambda expression that should return "True" if the passed node matches criteria.

            @return <AdvancedTag/None> - First match, or None

            @see getElementsCustomFilter
        '''

        for child in self.children:
            if filterFunc(child) is True:
                return child

            childSearchResult = child.getFirstElementCustomFilter(filterFunc)
            if childSearchResult is not None:
                return childSearchResult

        return None