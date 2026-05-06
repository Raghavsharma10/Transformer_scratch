def getElementsByTagName(self, tagName):
        '''
            getElementsByTagName - Gets elements within this collection having a specific tag name

            @param tagName - String of tag name

            @return - TagCollection of unique elements within this collection with given tag name
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret

        tagName = tagName.lower()
        _cmpFunc = lambda tag : bool(tag.tagName == tagName)
        
        for tag in self:
            TagCollection._subset(ret, _cmpFunc, tag)

        return ret