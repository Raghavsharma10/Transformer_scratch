def bind(self, alloy):
        '''
        Shallow copies this MethodParameter, and binds it to an alloy.
        This is required before calling.
        '''
        param = MethodParameter(self.name, self.method, self.dependencies,
                                self.units, self.aliases, self._references)
        param.alloy = alloy
        return param