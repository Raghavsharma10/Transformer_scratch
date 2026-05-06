def filterCollection(self, filterFunc):
        '''
            filterCollection - Filters only the immediate objects contained within this Collection against a function, not including any children

            @param filterFunc <function> - A function or lambda expression that returns True to have that element match

            @return TagCollection<AdvancedTag>
        '''
        ret = TagCollection()
        if len(self) == 0:
            return ret

        for tag in self:
            if filterFunc(tag) is True:
                ret.append(tag)

        return ret