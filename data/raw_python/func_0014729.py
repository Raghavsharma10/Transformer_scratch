def getElementsCustomFilter(self, filterFunc):
        '''
            getElementsCustomFilter - Get elements within this collection that match a user-provided function.

            @param filterFunc <function> - A function that returns True if the element matches criteria

            @return - TagCollection of all elements that matched criteria
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret

        _cmpFunc = lambda tag : filterFunc(tag) is True
        for tag in self:
            TagCollection._subset(ret, _cmpFunc, tag)

        return ret