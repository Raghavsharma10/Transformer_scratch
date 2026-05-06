def set_param(self, param, value):
        '''Set a parameter in this configuration set.'''
        self.data[param] = value
        self._object.configuration_data = utils.dict_to_nvlist(self.data)