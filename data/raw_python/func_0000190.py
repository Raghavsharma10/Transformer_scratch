def setOverrideValue(self, attributes, hostName):
        """ Function __setitem__
        Set a parameter of a foreman object as a dict

        @param key: The key to modify
        @param attribute: The data
        @return RETURN: The API result
        """
        self['override'] = True
        attrType = type(attributes)
        if attrType is dict:
            self['parameter_type'] = 'hash'
        elif attrType is list:
            self['parameter_type'] = 'array'
        else:
            self['parameter_type'] = 'string'
        orv = self.getOverrideValueForHost(hostName)
        if orv:
            orv['value'] = attributes
            return True
        else:
            return self.api.create('{}/{}/{}'.format(self.objName,
                                                     self.key,
                                                     'override_values'),
                                   {"override_value":
                                       {"match": "fqdn={}".format(hostName),
                                        "value": attributes}})