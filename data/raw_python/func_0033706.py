def get_parameter(self, name, default=None):
        '''
        Returns the named parameter if present, or the value of `default`,
        otherwise.
        '''
        if hasattr(self, name):
            item = getattr(self, name)
            if isinstance(item, Parameter):
                return item
        return default