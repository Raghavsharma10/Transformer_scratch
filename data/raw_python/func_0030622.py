def qs_delete(self, *keys):
        '''Delete value from QuerySet MultiDict'''
        query = self.query.copy()
        for key in set(keys):
            try:
                del query[key]
            except KeyError:
                pass
        return self._copy(query=query)