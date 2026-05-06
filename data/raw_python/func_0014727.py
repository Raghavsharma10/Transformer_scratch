def getElementsByAttr(self, attr, value):
        '''
            getElementsByAttr - Get elements within this collection posessing a given attribute/value pair

            @param attr - Attribute name (lowercase)
            @param value - Matching value

            @return - TagCollection of all elements matching name/value
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret

        attr = attr.lower()
        _cmpFunc = lambda tag : tag.getAttribute(attr) == value
        for tag in self:
            TagCollection._subset(ret, _cmpFunc, tag)
        
        return ret