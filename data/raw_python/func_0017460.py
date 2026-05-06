def oftype(self, typ):
        '''Return a generator of formatters codes of type typ'''
        for key, val in self.items():
            if val.type == typ:
                yield key