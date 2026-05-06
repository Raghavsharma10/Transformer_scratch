def set_conf_set_value(self, set_name, param, value):
        '''Set a configuration set parameter value.

        @param set_name The name of the configuration set the destination
                        parameter is in.
        @param param The name of the parameter to set.
        @param value The new value for the parameter.
        @raises NoSuchConfSetError, NoSuchConfParamError

        '''
        with self._mutex:
            if not set_name in self.conf_sets:
                raise exceptions.NoSuchConfSetError(set_name)
            if not self.conf_sets[set_name].has_param(param):
                raise exceptions.NoSuchConfParamError(param)
            self.conf_sets[set_name].set_param(param, value)
            self._conf.set_configuration_set_values(\
                    self.conf_sets[set_name].object)