def construct(self):
        '''Build the :class:`QueryElement` representing this query.'''
        if self.__construct is None:
            self.__construct = self._construct()
        return self.__construct