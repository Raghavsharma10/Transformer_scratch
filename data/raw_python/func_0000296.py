def getParam(self, name=None):
        """ Function getParam
        Return a dict of parameters or a parameter value

        @param key: The parameter name
        @return RETURN: dict of parameters or a parameter value
        """
        if 'parameters' in self.keys():
            l = {x['name']: x['value'] for x in self['parameters'].values()}
            if name:
                if name in l.keys():
                    return l[name]
                else:
                    return False
            else:
                return l