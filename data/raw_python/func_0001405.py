def find(self, tagtype, **kwargs):
        '''Get the first tag with a type in this token '''
        for t in self.__tags:
            if t.tagtype == tagtype:
                return t
        if 'default' in kwargs:
            return kwargs['default']
        else:
            raise LookupError("Token {} is not tagged with the speficied tagtype ({})".format(self, tagtype))