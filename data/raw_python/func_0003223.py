def createMatcher(cls, *args, **kwargs):
        '''
        :param _ismatch: user-defined function ismatch(event) for matching test
        :param \*args: indices
        :param \*\*kwargs: index_name=index_value for matching criteria
        '''
        if kwargs and not args:
            return EventMatcher(tuple(getattr(cls, ind) if ind[:10] == '_classname' else kwargs.get(ind) for ind in cls.indicesNames()), kwargs.get('_ismatch'))
        else:
            return EventMatcher(tuple(cls._generateIndices(args)), kwargs.get('_ismatch'))