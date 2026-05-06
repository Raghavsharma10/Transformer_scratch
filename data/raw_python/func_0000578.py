def getParamFromEnv(self, var, default=''):
        """ Function getParamFromEnv
        Search a parameter in the host environment

        @param var: the var name
        @param hostgroup: the hostgroup item linked to this host
        @param default: default value
        @return RETURN: the value
        """
        if self.getParam(var):
            return self.getParam(var)
        if self.hostgroup:
            if self.hostgroup.getParam(var):
                return self.hostgroup.getParam(var)
        if self.domain.getParam('password'):
            return self.domain.getParam('password')
        else:
            return default