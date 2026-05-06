def qs_set(self, *args, **kwargs):
        '''Set values in QuerySet MultiDict'''
        if args and kwargs:
            raise TypeError('Use positional args or keyword args not both')
        query = self.query.copy()
        if args:
            mdict = MultiDict(args[0])
            for k in mdict.keys():
                if k in query:
                    del query[k]
            for k, v in mdict.items():
                query.add(k, v)
        else:
            for k, v in kwargs.items():
                query[k] = v
        return self._copy(query=query)