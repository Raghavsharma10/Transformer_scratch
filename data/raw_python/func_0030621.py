def qs_add(self, *args, **kwargs):
        '''Add value to QuerySet MultiDict'''
        query = self.query.copy()
        if args:
            mdict = MultiDict(args[0])
            for k, v in mdict.items():
                query.add(k, v)
        for k, v in kwargs.items():
            query.add(k, v)
        return self._copy(query=query)