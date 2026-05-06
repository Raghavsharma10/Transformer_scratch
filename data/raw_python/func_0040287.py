def get_method(self, node):
        '''Given a particular node, check the visitor instance for methods
        mathing the computed methodnames (the function is a generator).

        Note that methods are cached at the class level.
        '''
        methods = self._methods
        for methodname in self.get_methodnames(node):
            if methodname in methods:
                return methods[methodname]
            else:
                cls = self.__class__
                method = getattr(cls, methodname, None)
                if method is not None:
                    methods[methodname] = method
                    return method