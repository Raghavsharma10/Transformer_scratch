def checkAndCreateParams(self, params):
        """ Function checkAndCreateParams
        Check and add global parameters

        @param key: The parameter name
        @param params: The params dict
        @return RETURN: boolean
        """
        actual_params = self['parameters'].keys()
        for k, v in params.items():
            if k not in actual_params:
                self['parameters'].append({"name": k, "value": v})
        self.reload()
        return self['parameters'].keys() == params.keys()