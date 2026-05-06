def find_all(self, tagtype):
        ''' Find all token-level tags with the specified tagtype '''
        return [t for t in self.__tags if t.tagtype == tagtype]