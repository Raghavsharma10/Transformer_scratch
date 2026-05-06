def getElementsByClassName(self, className):
        '''
            getElementsByClassName - Get elements within this collection containing a specific class name

            @param className - A single class name

            @return - TagCollection of unique elements within this collection tagged with a specific class name
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret
        _cmpFunc = lambda tag : tag.hasClass(className)
        for tag in self:
            TagCollection._subset(ret, _cmpFunc, tag)
        
        return ret