def set_config_parameter(self, param, value):
        '''Set a configuration parameter of the manager.

        @param The parameter to set.
        @value The new value for the parameter.
        @raises FailedToSetConfigurationError

        '''
        with self._mutex:
            if self._obj.set_configuration(param, value) != RTC.RTC_OK:
                raise exceptions.FailedToSetConfigurationError(param, value)
            # Force a reparse of the configuration
            self._configuration = None