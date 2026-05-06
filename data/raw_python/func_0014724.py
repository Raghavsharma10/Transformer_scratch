def getElementsByName(self, name):
        '''
            getElementsByName - Get elements within this collection having a specific name

            @param name - String of "name" attribute

            @return - TagCollection of unique elements within this collection with given "name"
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret
        _cmpFunc = lambda tag : bool(tag.name == name)
        for tag in self:
            TagCollection._subset(ret, _cmpFunc, tag)

        return ret