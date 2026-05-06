def getElementsWithAttrValues(self, attr, values):
        '''
            getElementsWithAttrValues - Get elements within this collection possessing an attribute name matching one of several values

            @param attr <lowercase str> - Attribute name (lowerase)
            @param values set<str> - Set of possible matching values

            @return - TagCollection of all elements matching criteria
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret

        if type(values) != set:
            values = set(values)

        attr = attr.lower()
        _cmpFunc = lambda tag : tag.getAttribute(attr) in values
        for tag in self:
            TagCollection._subset(ret, _cmpFunc, tag)
        
        return ret