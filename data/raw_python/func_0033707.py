def get_unique_parameters(self):
        '''
        Returns a list of the unique parameters (no duplicates).
        '''
        # start with parameters in the `_parameters` dictionary
        parameters = self._parameters.values()
        # add parameters defined with the class
        for name in dir(self):
            item = getattr(self, name)
            if isinstance(item, Parameter):
                if item.name not in self._parameters:
                    parameters.append(item)
        return parameters