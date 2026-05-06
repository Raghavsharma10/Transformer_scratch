def get_tag(self, tagtype):
        ''' Get the first tag of a particular type'''
        for tag in self.__tags:
            if tag.tagtype == tagtype:
                return tag
        return None