def getParentElementCustomFilter(self, filterFunc):
        '''
            getParentElementCustomFilter - Runs through parent on up to document root, returning the

                                              first tag which filterFunc(tag) returns True.

                @param filterFunc <function/lambda> - A function or lambda expression that should return "True" if the passed node matches criteria.

                @return <AdvancedTag/None> - First match, or None


                @see getFirstElementCustomFilter for matches against children
        '''
        parentNode = self.parentNode
        while parentNode:

            if filterFunc(parentNode) is True:
                return parentNode

            parentNode = parentNode.parentNode

        return None