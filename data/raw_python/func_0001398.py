def get_tags(self, tagtype):
        ''' Get all tags of a type '''
        return [t for t in self.__tags if t.tagtype == tagtype]