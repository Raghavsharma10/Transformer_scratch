def name(self, name=None):
        '''api name, default is module.__name__'''
        if name:
            self._name = name
            return self
        return self._name